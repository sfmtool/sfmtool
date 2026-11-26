// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Photographs for the panels and the 3D view's camera-view background,
//! read off the GUI thread.
//!
//! A panel draws every frame and must not wait for a JPEG decode, which takes
//! about 40 ms for a 4K photograph. So what it calls on the GUI thread only
//! looks in the photograph cache, and on a miss starts one task on the rayon
//! pool, which asks egui for a repaint when it ends; the next frame's look
//! finds what the task made.
//!
//! - Image Detail draws photograph tiles. [`display_layout`] answers the
//!   photograph's layout ([`PhotographCache::layout`]), starting one
//!   [`PhotographCache::request_layout`] when it is not known, and
//!   [`display_tiles`] answers the tiles asked for
//!   ([`PhotographCache::peek_tile`]), starting one
//!   [`PhotographCache::try_decode_tiles`] for the ones missing. At most one
//!   task of each kind is in flight per photograph, and the tiles a decode
//!   task answers are kept here until a panel asks for them, which is how a
//!   cache of budget 0, which keeps nothing, hands them over.
//!   [`prefetch_tiles`] starts the same decode a frame ahead, for the image
//!   that animation shows next.
//! - The 3D view's camera-view background draws one level whole.
//!   [`display_level`] answers it, starting one task that asks for the
//!   level's tiles with [`PhotographCache::try_decode_tiles`] and builds the
//!   level from the tiles that call answers, at every budget, so the task
//!   never waits on a slot or on the build count. It shares the tile
//!   decodes' limit of one task per photograph and their remembered
//!   failures, and keeps the one level it last answered until the background
//!   asks for it.
//! - Track View, the Track Table's crops and Measurement View's endpoint
//!   crops render their pictures from photograph tiles. [`read_peeked`]
//!   runs a render on a photograph made of exactly the tiles it needs
//!   ([`TiledPhotograph::from_tiles`]), held for the render, once
//!   [`display_tile_set`] has every one of them, which it asks for together:
//!   all of them handed over at once or none, the missing ones decoded by
//!   one [`PhotographCache::try_decode_tiles`] as Image Detail's are.
//!   [`read_peeked_without_asking`] runs the same render on what is already
//!   there and asks for nothing, for the Track Table's rows while they
//!   scroll.
//!
//! No panel reads a whole pyramid, and none decodes on the GUI thread, at
//! any budget. None makes a mipmap: an out-of-core photograph whose mipmap
//! is not current is [`DisplayLayout::Building`], queued for the *Build
//! mipmaps* quiet task ([`super::build_mipmaps`]), and none of its tiles is
//! asked for until that task has made it.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use sfmtool_core::camera::image::ImageU8;
use sfmtool_core::camera::GetError;
use sfmtool_core::camera::{
    CoarsestLevel, PeekedLayout, PeekedTile, PhotographCache, PhotographReadError, PhotographTile,
    PyramidLayout, TileError, TileKey, TiledPhotograph, TryDecode,
};
use sfmtool_core::progress::Progress;
use sfmtool_core::workspace::WorkspaceConfigError;

use super::build_mipmaps::MipmapList;

#[cfg(test)]
mod tests;

/// What a panel knows of a photograph's layout on this frame.
#[derive(Clone, Debug)]
pub(crate) enum DisplayLayout {
    /// Its level sizes and tile grid: its tiles may be asked for.
    Known(Arc<PyramidLayout>),
    /// Its header is being read on a worker, or about to be. A repaint
    /// follows the read.
    Loading,
    /// An out-of-core photograph whose mipmap is missing or stale: queued
    /// for *Build mipmaps*, whose run that makes it ends with a repaint. No
    /// tile of it is asked for meanwhile. It carries the coarsest level once
    /// the build making the mipmap has published it, which Image Detail and
    /// the 3D background draw while they wait; the run repaints when it
    /// lands ([`sfmtool_core::camera::COARSEST_PUBLISHED`]).
    Building(Option<Arc<CoarsestLevel>>),
    /// The file could not be read or decoded.
    Unreadable,
    /// The photograph's workspace file cannot be read, or its
    /// `mipmap_options` is invalid, so the cache refuses every photograph of
    /// that workspace ([`GetError::Workspace`]); the error names the file
    /// and the key. Image Detail says so; the other panels draw it as they
    /// draw [`DisplayLayout::Unreadable`].
    WorkspaceFileError(WorkspaceConfigError),
}

/// What a panel has for one photograph tile on this frame.
#[derive(Clone)]
pub(crate) enum DisplayTile {
    /// The tile, with its border.
    Ready(Arc<PhotographTile>),
    /// Being decoded on a worker, or waiting to be asked for again because
    /// the decode could not start. A repaint follows either.
    Loading,
    /// The photograph could not be read or decoded.
    Unreadable,
}

/// What a panel has of the photograph tiles one render needs on this frame.
pub(crate) enum DisplayTileSet {
    /// Every tile asked for, in the order of the keys.
    Ready(Vec<Arc<PhotographTile>>),
    /// Some of them are being decoded on a worker, or wait to be asked for
    /// again because the decode could not start. A repaint follows either.
    Loading,
    /// The photograph could not be read or decoded.
    Unreadable,
}

/// What a panel says of a photograph that cannot be read or decoded, after
/// "The photograph could not be read: " or a sentence like it.
pub(crate) const UNREADABLE_PHOTOGRAPH: &str = "the file could not be read or decoded";

/// What a panel says, with a spinner, in place of a picture of an
/// out-of-core photograph whose mipmap *Build mipmaps* is making
/// ([`DisplayLayout::Building`]).
pub(crate) const BUILDING_MIPMAP_LABEL: &str = "Building mipmap…";

/// What a panel's render from its photograph's tiles gave on this frame.
pub(crate) enum PeekedRead<T> {
    /// The render ran on a photograph made of the tiles it needs.
    Read(Result<T, PhotographReadError>),
    /// The layout or some of the tiles are being read on a worker. A repaint
    /// follows.
    Loading,
    /// An out-of-core photograph whose mipmap is missing or stale
    /// ([`DisplayLayout::Building`]): none of its tiles is asked for, and
    /// the picture draws a spinner with [`BUILDING_MIPMAP_LABEL`] until the
    /// run of *Build mipmaps* that makes the mipmap ends, with a repaint.
    Building,
    /// The photograph could not be read or decoded, or its workspace file
    /// refuses it.
    Unreadable,
}

/// What the 3D view's camera-view background has for one level of a
/// photograph on this frame.
#[derive(Clone)]
pub(crate) enum DisplayLevel {
    /// The level whole, of the size the layout gives it.
    Ready(Arc<ImageU8>),
    /// Being read on a worker, or waiting to be asked for again because the
    /// decode could not start. A repaint follows either.
    Loading,
    /// The photograph could not be read or decoded.
    Unreadable,
}

/// One photograph as a panel asks for it: the cache that holds it, the
/// requests in flight for it, and its path.
#[derive(Clone, Copy)]
pub(crate) struct PhotographSource<'a> {
    pub(crate) cache: &'a Arc<PhotographCache>,
    pub(crate) requests: &'a PhotographRequests,
    pub(crate) path: &'a Path,
}

/// How long after a decode task answered `Busy` its panel asks again: the
/// decode did not fit under the build budget, or another caller is decoding
/// the photograph, and a repaint at once would only ask again while that
/// lasts.
const BUSY_RETRY: Duration = Duration::from_millis(50);

/// The most photographs whose answers are kept for the panels to take, the
/// oldest dropped first, so the answers of photographs no panel shows any
/// more are not kept for the session. A photograph's kept tiles are at most
/// [`MAX_KEPT_TILES`], or one decode's answer when that is more: at most 128
/// tiles of 514 x 514 RGB, about 101 MB, in all, when no one render needs
/// more than 32.
const MAX_ANSWERED_PHOTOGRAPHS: usize = 4;

/// The most answered tiles kept for one photograph: an answer that would
/// take the photograph's kept tiles past it replaces them. Image Detail asks
/// for at most 16 tiles of a photograph at once (`image_detail::photograph_tiles`'s
/// upload cap) and every ask of its drops the kept tiles it does not ask for,
/// so its own answers, the kept and the new, are at most 32. The pictures
/// [`read_peeked`] renders (Track View's, the Track Table's crops and
/// Measurement View's endpoint crops) ask for the tiles of one render
/// together ([`display_tile_set`]) and drop no other ask's, since several
/// pictures read one photograph, so their answers for the photograph build
/// up until the renders take them; this bounds them.
const MAX_KEPT_TILES: usize = 32;

/// A file's length and modification time, which tell a file that changed
/// since a decode of it failed: `None` when it cannot be read.
type FileStamp = Option<(u64, Option<std::time::SystemTime>)>;

/// `path`'s [`FileStamp`] now.
fn file_stamp(path: &Path) -> FileStamp {
    std::fs::metadata(path)
        .ok()
        .map(|m| (m.len(), m.modified().ok()))
}

/// A photograph a read, a decode or a build found could not be read.
struct Failure {
    /// The file whose change makes the photograph worth asking for again:
    /// the photograph's own, or the workspace file that refused it.
    watched: PathBuf,
    /// The stamp `watched` had then.
    stamp: FileStamp,
    /// The workspace file's error, when the photograph's workspace file
    /// refused it ([`GetError::Workspace`]), which Image Detail says.
    workspace: Option<WorkspaceConfigError>,
}

/// `path`'s refusal by its workspace file, when the layout the cache
/// answers for it after a read is one: the file's error, and the stamp the
/// workspace file had before a second read that found the refusal still
/// there. Taking the stamp before that read, rather than after the first,
/// means a file fixed while the first read ran is not kept as refused with
/// the fixed file's stamp, which no later check would find changed. Reads
/// the header again, so it runs off the GUI thread.
fn confirmed_refusal(
    cache: &PhotographCache,
    path: &Path,
) -> Option<(WorkspaceConfigError, FileStamp)> {
    let PeekedLayout::Unreadable(GetError::Workspace(e)) = cache.layout(path) else {
        return None;
    };
    let stamp = file_stamp(workspace_file(&e));
    cache.request_layout(path);
    match cache.layout(path) {
        PeekedLayout::Unreadable(GetError::Workspace(e)) => Some((e, stamp)),
        _ => None,
    }
}

/// For a decode its workspace file refused, that refusal as a second read
/// of the header confirms it ([`confirmed_refusal`]), or `None` when that
/// read no longer finds it, as when the file was fixed while the decode
/// ran. Reads the header only for such a refusal, so it runs off the GUI
/// thread.
fn decode_refusal<T>(
    cache: &PhotographCache,
    path: &Path,
    answer: &Result<T, TileError>,
) -> Option<(WorkspaceConfigError, FileStamp)> {
    if !matches!(
        answer,
        Err(TileError::Unreadable {
            why: GetError::Workspace(_),
            ..
        })
    ) {
        return None;
    }
    cache.request_layout(path);
    confirmed_refusal(cache, path)
}

/// The workspace file `e` is an error of.
fn workspace_file(e: &WorkspaceConfigError) -> &Path {
    match e {
        WorkspaceConfigError::Io { path, .. }
        | WorkspaceConfigError::Json { path, .. }
        | WorkspaceConfigError::MipmapOptions { path, .. }
        | WorkspaceConfigError::MipmapOptionsPresent { path }
        | WorkspaceConfigError::Version { path, .. } => path,
    }
}

/// A task to run off the GUI thread.
type Task = Box<dyn FnOnce() + Send>;

/// Where tasks are started: the rayon pool, or in the tests a queue the test
/// runs by hand.
type Spawner = Arc<dyn Fn(Task) + Send + Sync>;

/// A test's hook told the chunk each run of *Build mipmaps* takes, as it
/// starts.
#[cfg(test)]
pub(crate) type ChunkHook = Arc<dyn Fn(&[PathBuf]) + Send + Sync>;

/// The tiles a photograph's decode tasks answered that no panel has taken
/// yet.
type Answered = HashMap<TileKey, Arc<PhotographTile>>;

/// The level a background task answered, kept until the background asks for
/// it.
struct AnsweredLevel {
    path: PathBuf,
    level: usize,
    pixels: Arc<ImageU8>,
}

/// The tile and level decodes in flight and what they answered.
#[derive(Default)]
struct TileRequests {
    /// The paths a decode task is running for, a tile decode's or a level's.
    in_flight: HashSet<PathBuf>,
    /// What the tile decodes answered, oldest first, at most
    /// [`MAX_ANSWERED_PHOTOGRAPHS`] paths.
    answered: Vec<(PathBuf, Answered)>,
    /// Every photograph a decode found could not be read or decoded, with the
    /// stamp its file had then. A file whose stamp has changed since is asked
    /// for again. Not bounded, unlike [`TileRequests::answered`]: an entry is
    /// a path and a stamp, and a failure dropped to make room would be
    /// decoded again on the next frame that shows the photograph, whose
    /// failure would drop another, so panels showing more such photographs
    /// than a bound would decode them in turn and repaint for ever. The cache
    /// remembers no failure under a budget of 0, so this is then the only
    /// memory of one.
    failures: HashMap<PathBuf, Failure>,
    /// The level the last level task answered and the background has not
    /// taken: one level, since one background is drawn.
    level: Option<AnsweredLevel>,
}

impl TileRequests {
    fn answered(&mut self, path: &Path) -> Option<&mut Answered> {
        self.answered
            .iter_mut()
            .find(|(p, _)| p == path)
            .map(|(_, answered)| answered)
    }

    /// Whether `path` failed.
    fn has_failed(&self, path: &Path) -> bool {
        self.failures.contains_key(path)
    }

    /// The file `path`'s failure watches and the stamp it had then, and
    /// whether the photograph's workspace file refused it, if it failed.
    fn failed(&self, path: &Path) -> Option<(PathBuf, FileStamp, bool)> {
        self.failures.get(path).map(|failure| {
            (
                failure.watched.clone(),
                failure.stamp,
                failure.workspace.is_some(),
            )
        })
    }

    /// Remember that `path` could not be read, in place of any tiles answered
    /// for it, with the photograph's stamp now, so it is asked for again once
    /// the photograph changes. Stats the file, so it runs off the GUI thread.
    fn remember_failure(&mut self, path: &Path) {
        let stamp = file_stamp(path);
        self.remember(path, None, stamp);
    }

    /// Remember that `path`'s workspace file refuses it with `e`, the file's
    /// stamp having been `stamp` before the read that found the refusal
    /// ([`confirmed_refusal`]), so a file fixed during that read is found
    /// changed by the next check.
    fn remember_refusal(&mut self, path: &Path, e: WorkspaceConfigError, stamp: FileStamp) {
        self.remember(path, Some(e), stamp);
    }

    /// Keep `path`'s failure, watching the workspace file `workspace` names
    /// or else the photograph, whose stamp was `stamp`, in place of any
    /// tiles answered for it.
    fn remember(&mut self, path: &Path, workspace: Option<WorkspaceConfigError>, stamp: FileStamp) {
        self.answered.retain(|(p, _)| p != path);
        let watched = workspace
            .as_ref()
            .map_or(path, workspace_file)
            .to_path_buf();
        self.failures.insert(
            path.to_path_buf(),
            Failure {
                watched,
                stamp,
                workspace,
            },
        );
    }

    /// Keep what a decode of `path` that failed with `e` tells later frames.
    /// A cancelled decode tells nothing, and the next frame asks again. An
    /// unreadable photograph is remembered with its file's stamp, in place of
    /// any tiles answered for it. One its workspace file refused is
    /// remembered with that file's error and stamp when `refusal`, a second
    /// read's ([`decode_refusal`]), still finds the refusal, and otherwise
    /// not at all, so the next frame asks again. A photograph whose mipmap
    /// was found not current (`Changed`) is `Building` in the cache from then
    /// on, so the next frame asks for none of its tiles and queues it for
    /// *Build mipmaps*; the tiles answered for it are dropped, since they may
    /// be of the files that changed. A key or a level the photograph does not have
    /// comes from a layout read before the file changed size: the cache
    /// forgets the path, so the next frame reads the header again rather than
    /// ask for the same pixels on every frame.
    fn failed_with(
        &mut self,
        cache: &PhotographCache,
        path: &Path,
        e: TileError,
        refusal: Option<(WorkspaceConfigError, FileStamp)>,
    ) {
        match e {
            TileError::Cancelled => {}
            TileError::Unreadable {
                why: GetError::Workspace(_),
                ..
            } => {
                if let Some((e, stamp)) = refusal {
                    self.remember_refusal(path, e, stamp);
                }
            }
            TileError::Unreadable { .. } => self.remember_failure(path),
            TileError::Changed { .. } => self.answered.retain(|(p, _)| p != path),
            TileError::OutOfRange { .. } => cache.forget(path),
        }
    }

    /// Drop `path`'s failure, for a file that has changed since.
    fn forget_failure(&mut self, path: &Path) {
        self.failures.remove(path);
    }

    /// Drop `path`'s kept tiles that are not among `keys`, the ones a panel
    /// asks for now: tiles a decode answered for a view that has moved on.
    fn keep_only(&mut self, path: &Path, keys: &[TileKey]) {
        let Some(index) = self.answered.iter().position(|(p, _)| p == path) else {
            return;
        };
        let tiles = &mut self.answered[index].1;
        tiles.retain(|key, _| keys.contains(key));
        if tiles.is_empty() {
            self.answered.remove(index);
        }
    }

    /// How many answered tiles are kept, of every photograph.
    #[cfg(test)]
    fn kept_tiles(&self) -> usize {
        self.answered.iter().map(|(_, tiles)| tiles.len()).sum()
    }

    /// Whether `key` of `path` has been answered and not yet taken.
    fn holds(&mut self, path: &Path, key: TileKey) -> bool {
        self.answered(path)
            .is_some_and(|tiles| tiles.contains_key(&key))
    }

    /// `key` of `path` if a task answered it, left kept.
    fn kept(&mut self, path: &Path, key: TileKey) -> Option<Arc<PhotographTile>> {
        self.answered(path)?.get(&key).cloned()
    }

    /// Take `key` of `path` if a task answered it.
    fn take(&mut self, path: &Path, key: TileKey) -> Option<Arc<PhotographTile>> {
        let index = self.answered.iter().position(|(p, _)| p == path)?;
        let tiles = &mut self.answered[index].1;
        let tile = tiles.remove(&key);
        if tiles.is_empty() {
            self.answered.remove(index);
        }
        tile
    }

    /// Keep the tiles a task answered for `path` as its newest answer,
    /// dropping the oldest past [`MAX_ANSWERED_PHOTOGRAPHS`], and in place of
    /// the tiles kept before it when the two would be more than
    /// [`MAX_KEPT_TILES`]. A photograph that answered tiles has no failure.
    fn keep(&mut self, path: PathBuf, more: Answered) {
        self.failures.remove(&path);
        let answer = match self.answered.iter().position(|(p, _)| *p == path) {
            Some(index) => {
                let mut tiles = self.answered.remove(index).1;
                tiles.extend(more.iter().map(|(&key, tile)| (key, Arc::clone(tile))));
                if tiles.len() > MAX_KEPT_TILES {
                    more
                } else {
                    tiles
                }
            }
            None => more,
        };
        self.answered.push((path, answer));
        if self.answered.len() > MAX_ANSWERED_PHOTOGRAPHS {
            self.answered.remove(0);
        }
    }
}

/// The reads a panel has started and has not seen end, so each is asked for
/// once however many frames look at it meanwhile, and the tiles they answered.
#[derive(Clone)]
pub(crate) struct PhotographRequests {
    /// The paths a header read is running for.
    layouts_in_flight: Arc<Mutex<HashSet<PathBuf>>>,
    tiles: Arc<Mutex<TileRequests>>,
    /// The photographs queued for *Build mipmaps*, shared with its job.
    pub(super) mipmaps: Arc<Mutex<MipmapList>>,
    spawn: Spawner,
    /// Told the chunk each run of *Build mipmaps* takes, as it starts.
    #[cfg(test)]
    on_chunk: Option<ChunkHook>,
}

impl Default for PhotographRequests {
    /// Requests whose tasks run on the rayon pool.
    fn default() -> Self {
        Self::with_spawner(Arc::new(|task: Task| rayon::spawn(task)))
    }
}

impl std::fmt::Debug for PhotographRequests {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhotographRequests").finish_non_exhaustive()
    }
}

/// Lock `mutex`, ignoring a poison: nothing here panics while one is held.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

impl PhotographRequests {
    fn with_spawner(spawn: Spawner) -> Self {
        Self {
            layouts_in_flight: Arc::default(),
            tiles: Arc::default(),
            mipmaps: Arc::default(),
            spawn,
            #[cfg(test)]
            on_chunk: None,
        }
    }

    /// Keep `path`, whose mipmap *Build mipmaps* could not make, as a failed
    /// read is kept: answered as unreadable, with its file's stamp, until
    /// the file changes, so a cache that remembers no failure is not asked
    /// to make it again on every frame. A build its workspace file refused,
    /// as when the file broke after the photograph was queued, is found so
    /// by reading the header again, since `get_many` answers no error and a
    /// cache of budget 0 keeps none, and is kept as the header read keeps
    /// one ([`confirmed_refusal`]): a workspace file error, watching that
    /// file. Runs on the task's thread.
    pub(super) fn failed_to_build(&self, cache: &PhotographCache, path: &Path) {
        cache.request_layout(path);
        let refusal = confirmed_refusal(cache, path);
        let mut tiles = lock(&self.tiles);
        match refusal {
            Some((e, stamp)) => tiles.remember_refusal(path, e, stamp),
            None => tiles.remember_failure(path),
        }
    }

    /// Start `task` where these requests start theirs: on the rayon pool, or
    /// in the tests on the queue `Self::queued` holds back.
    pub(super) fn spawn(&self, task: impl FnOnce() + Send + 'static) {
        (self.spawn)(Box::new(task));
    }

    /// Tell the test's hook the chunk a run of *Build mipmaps* has taken.
    #[cfg(test)]
    pub(super) fn chunk_started(&self, chunk: &[PathBuf]) {
        if let Some(on_chunk) = &self.on_chunk {
            on_chunk(chunk);
        }
    }

    /// Call `hook` with the chunk each run of *Build mipmaps* takes, as it
    /// starts: what lets a test hold a run while it asks for a task.
    #[cfg(test)]
    pub(crate) fn set_chunk_hook(&mut self, hook: ChunkHook) {
        self.on_chunk = Some(hook);
    }

    /// The paths waiting in *Build mipmaps*' list, the next first.
    #[cfg(test)]
    pub(crate) fn mipmaps_waiting(&self) -> Vec<PathBuf> {
        super::build_mipmaps::lock(&self.mipmaps).waiting()
    }

    /// Start reading `path`'s header into `cache` unless a read of it is
    /// already running, and repaint `ctx` when it ends. A photograph its
    /// workspace file refuses is remembered as a failure watching that file
    /// ([`TileRequests::remember_refusal`]), since the cache answers its
    /// layout so, without a stat, until the path is forgotten.
    fn request_layout(&self, cache: &Arc<PhotographCache>, path: &Path, ctx: &egui::Context) {
        if !lock(&self.layouts_in_flight).insert(path.to_path_buf()) {
            return;
        }
        let cache = Arc::clone(cache);
        let in_flight = Arc::clone(&self.layouts_in_flight);
        let tiles = Arc::clone(&self.tiles);
        let path = path.to_path_buf();
        let ctx = ctx.clone();
        (self.spawn)(Box::new(move || {
            cache.request_layout(&path);
            if let Some((e, stamp)) = confirmed_refusal(&cache, &path) {
                lock(&tiles).remember_refusal(&path, e, stamp);
            }
            lock(&in_flight).remove(&path);
            ctx.request_repaint();
        }));
    }

    /// Start one decode of `keys` of `path` unless one is already running for
    /// `path`, keep the tiles it answers, and repaint `ctx` when it ends: at
    /// once, or after [`BUSY_RETRY`] when it answered `Busy`.
    fn decode_tiles(
        &self,
        cache: &Arc<PhotographCache>,
        path: &Path,
        keys: Vec<TileKey>,
        ctx: &egui::Context,
    ) {
        if !lock(&self.tiles).in_flight.insert(path.to_path_buf()) {
            return;
        }
        let cache = Arc::clone(cache);
        let tiles = Arc::clone(&self.tiles);
        let path = path.to_path_buf();
        let ctx = ctx.clone();
        (self.spawn)(Box::new(move || {
            let answer = cache.try_decode_tiles(&path, &keys, &Progress::none());
            let busy = matches!(answer, Ok(TryDecode::Busy));
            let refusal = decode_refusal(&cache, &path, &answer);
            {
                // One lock for the answer and the end of the flight, so a
                // frame that finds the path no longer in flight finds its
                // answer.
                let mut tiles = lock(&tiles);
                match answer {
                    Ok(TryDecode::Decoded(found)) => {
                        let found = found.into_iter().map(|tile| (tile.key, tile)).collect();
                        tiles.keep(path.clone(), found);
                    }
                    // Nothing was decoded, and the next frame asks again.
                    Ok(TryDecode::Busy) => {}
                    Err(e) => tiles.failed_with(&cache, &path, e, refusal),
                }
                tiles.in_flight.remove(&path);
            }
            if busy {
                ctx.request_repaint_after(BUSY_RETRY);
            } else {
                ctx.request_repaint();
            }
        }));
    }

    /// Start one task that reads level `level` of `path`, whose layout is
    /// `layout`, unless a decode is already running for `path`: it asks for
    /// the level's tiles with [`PhotographCache::try_decode_tiles`] and
    /// builds the level from the tiles that call answers. The level is kept
    /// for [`display_level`], and `ctx` repainted when the task ends: at once, or
    /// after [`BUSY_RETRY`] when the decode answered `Busy`.
    fn decode_level(
        &self,
        cache: &Arc<PhotographCache>,
        path: &Path,
        layout: &Arc<PyramidLayout>,
        level: usize,
        ctx: &egui::Context,
    ) {
        if !lock(&self.tiles).in_flight.insert(path.to_path_buf()) {
            return;
        }
        let cache = Arc::clone(cache);
        let tiles = Arc::clone(&self.tiles);
        let path = path.to_path_buf();
        let layout = Arc::clone(layout);
        let ctx = ctx.clone();
        (self.spawn)(Box::new(move || {
            let answer = read_level(&cache, &path, &layout, level);
            let busy = matches!(answer, Ok(None));
            let refusal = decode_refusal(&cache, &path, &answer);
            {
                // One lock for the answer and the end of the flight, so a
                // frame that finds the path no longer in flight finds its
                // answer.
                let mut tiles = lock(&tiles);
                match answer {
                    Ok(Some(pixels)) => {
                        tiles.level = Some(AnsweredLevel {
                            path: path.clone(),
                            level,
                            pixels,
                        });
                    }
                    // Nothing was decoded, and the next frame asks again.
                    Ok(None) => {}
                    Err(e) => tiles.failed_with(&cache, &path, e, refusal),
                }
                tiles.in_flight.remove(&path);
            }
            if busy {
                ctx.request_repaint_after(BUSY_RETRY);
            } else {
                ctx.request_repaint();
            }
        }));
    }

    /// Answer whether `source`'s photograph failed, and if it did, start one
    /// check on the rayon pool of whether the file its failure watches has
    /// changed since, which drops the failure and repaints when it has, so a
    /// file that is fixed loads again: the photograph's own, or the
    /// workspace file that refused it, in which case the cache forgets the
    /// path too, since it answers the refusal without a stat until then. The
    /// check stats the file, off this thread.
    fn failed(&self, source: PhotographSource<'_>, ctx: &egui::Context) -> bool {
        let path = source.path;
        let (watched, stamp, workspace) = {
            let mut tiles = lock(&self.tiles);
            let Some(failed) = tiles.failed(path) else {
                return false;
            };
            if !tiles.in_flight.insert(path.to_path_buf()) {
                return true;
            }
            failed
        };
        let tiles = Arc::clone(&self.tiles);
        let cache = Arc::clone(source.cache);
        let path = path.to_path_buf();
        let ctx = ctx.clone();
        (self.spawn)(Box::new(move || {
            let changed = file_stamp(&watched) != stamp;
            {
                let mut tiles = lock(&tiles);
                if changed {
                    tiles.forget_failure(&path);
                    if workspace {
                        cache.forget(&path);
                    }
                }
                tiles.in_flight.remove(&path);
            }
            if changed {
                ctx.request_repaint();
            }
        }));
        true
    }

    /// [`Self::failed`] with `ctx`; with none, only whether `source`'s
    /// photograph failed, starting no check of the file.
    fn failed_asking(&self, source: PhotographSource<'_>, ctx: Option<&egui::Context>) -> bool {
        match ctx {
            Some(ctx) => self.failed(source, ctx),
            None => lock(&self.tiles).has_failed(source.path),
        }
    }

    /// The workspace file's error that a failed decode of `path` was
    /// refused with, if it was.
    fn workspace_refusal(&self, path: &Path) -> Option<WorkspaceConfigError> {
        lock(&self.tiles)
            .failures
            .get(path)
            .and_then(|failure| failure.workspace.clone())
    }

    /// How many answered tiles are kept, of every photograph.
    #[cfg(test)]
    pub(crate) fn kept_tiles(&self) -> usize {
        lock(&self.tiles).kept_tiles()
    }

    /// Whether an answered level is kept.
    #[cfg(test)]
    pub(crate) fn keeps_a_level(&self) -> bool {
        lock(&self.tiles).level.is_some()
    }

    /// Requests whose tasks wait in the queue answered beside them until a
    /// test runs them, so a test steps the tasks as it steps frames.
    #[cfg(test)]
    pub(crate) fn queued() -> (Self, TaskQueue) {
        let queue = TaskQueue::default();
        let tasks = Arc::clone(&queue.0);
        let requests = Self::with_spawner(Arc::new(move |task: Task| lock(&tasks).push(task)));
        (requests, queue)
    }
}

/// The tasks [`PhotographRequests::queued`] holds back, in the order they
/// were started.
#[cfg(test)]
#[derive(Clone, Default)]
pub(crate) struct TaskQueue(Arc<Mutex<Vec<Task>>>);

#[cfg(test)]
impl TaskQueue {
    /// How many tasks wait.
    pub(crate) fn len(&self) -> usize {
        lock(&self.0).len()
    }

    /// Run every waiting task, in order, on a thread of its own, and wait for
    /// them with a time limit, so a task that never returns fails the test
    /// rather than hang it. Answers how many ran.
    pub(crate) fn run_off_this_thread(&self) -> usize {
        let tasks = std::mem::take(&mut *lock(&self.0));
        let count = tasks.len();
        let (done, finished) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for task in tasks {
                task();
            }
            let _ = done.send(());
        });
        finished
            .recv_timeout(Duration::from_secs(60))
            .expect("the tasks ended within the time limit");
        count
    }
}

/// The layout of `source`'s photograph, starting one read of its header on
/// the rayon pool when it is not known. Reads nothing on this thread, at any
/// budget, and allocates nothing for a layout the cache already knows.
///
/// A photograph whose last decode task failed is answered as unreadable,
/// which is what a cache of budget 0, which remembers no failure, would
/// otherwise never say, until its file changes. One its workspace file
/// refuses, found so by the header read or by a decode, is
/// [`DisplayLayout::WorkspaceFileError`] with the file's error.
///
/// An out-of-core photograph whose mipmap is not current is
/// [`DisplayLayout::Building`], and is added to the front of *Build
/// mipmaps*' list once while it waits there ([`super::build_mipmaps`]); a
/// panel never makes a mipmap, and asks for none of its tiles meanwhile.
pub(crate) fn display_layout(source: PhotographSource<'_>, ctx: &egui::Context) -> DisplayLayout {
    layout_of(source, Some(ctx))
}

/// [`display_layout`], which with no `ctx` asks for nothing: it neither
/// starts a header read, nor checks whether a file that failed has changed,
/// nor queues a photograph for *Build mipmaps*.
fn layout_of(source: PhotographSource<'_>, ctx: Option<&egui::Context>) -> DisplayLayout {
    if source.requests.failed_asking(source, ctx) {
        return match source.requests.workspace_refusal(source.path) {
            Some(e) => DisplayLayout::WorkspaceFileError(e),
            None => DisplayLayout::Unreadable,
        };
    }
    match source.cache.layout(source.path) {
        PeekedLayout::Known(layout) => DisplayLayout::Known(layout),
        PeekedLayout::Building { coarsest } => {
            if ctx.is_some() {
                super::build_mipmaps::lock(&source.requests.mipmaps).queue(source.path);
            }
            DisplayLayout::Building(coarsest)
        }
        PeekedLayout::Unreadable(GetError::Workspace(e)) => DisplayLayout::WorkspaceFileError(e),
        PeekedLayout::Unreadable(_) => DisplayLayout::Unreadable,
        PeekedLayout::NotKnown => {
            if let Some(ctx) = ctx {
                source
                    .requests
                    .request_layout(source.cache, source.path, ctx);
            }
            DisplayLayout::Loading
        }
    }
}

/// The tiles `keys` of `source`'s photograph, in the order of the keys,
/// starting one decode on the rayon pool for the ones the cache does not
/// hold, unless one is already running for the photograph. Never decodes on
/// this thread, at any budget.
///
/// A tile a decode task answered is handed over once: the caller keeps it,
/// as Image Detail keeps it in a texture, and asks for it again only after
/// dropping it. So a caller asks only for what it will use this frame, since
/// under a cache of budget 0 a tile it is handed and drops is decoded again.
/// The photograph's kept tiles the call does not ask for are dropped, since
/// they were answered for a view that has moved on.
pub(crate) fn display_tiles(
    source: PhotographSource<'_>,
    keys: &[TileKey],
    ctx: &egui::Context,
) -> Vec<DisplayTile> {
    if source.requests.failed(source, ctx) {
        return vec![DisplayTile::Unreadable; keys.len()];
    }
    let taken: Vec<Option<Arc<PhotographTile>>> = {
        let mut tiles = lock(&source.requests.tiles);
        let taken = keys
            .iter()
            .map(|&key| tiles.take(source.path, key))
            .collect();
        tiles.keep_only(source.path, keys);
        taken
    };
    let mut missing = Vec::new();
    let answers = keys
        .iter()
        .zip(taken)
        .map(|(&key, taken)| match taken {
            Some(tile) => DisplayTile::Ready(tile),
            None => match source.cache.peek_tile(source.path, key) {
                PeekedTile::Ready(tile) => DisplayTile::Ready(tile),
                PeekedTile::Unreadable => DisplayTile::Unreadable,
                // Asked for by no task: the next frame's layout is
                // `Building`, which queues the photograph.
                PeekedTile::Building => DisplayTile::Loading,
                PeekedTile::NotReady => {
                    missing.push(key);
                    DisplayTile::Loading
                }
            },
        })
        .collect();
    if !missing.is_empty() {
        source
            .requests
            .decode_tiles(source.cache, source.path, missing, ctx);
    }
    answers
}

/// Ask for the tiles `keys` of `source`'s photograph ahead of the frame that
/// draws them: one decode on the rayon pool for the ones no task has answered,
/// unless one is already running for the photograph, its tiles kept for
/// [`display_tiles`] to hand over. Peeks at nothing, so it copies no pixels on
/// this thread, and once every key is answered it starts nothing. The
/// photograph's kept tiles it does not ask for are dropped.
pub(crate) fn prefetch_tiles(source: PhotographSource<'_>, keys: &[TileKey], ctx: &egui::Context) {
    if source.requests.failed(source, ctx) {
        return;
    }
    let missing: Vec<TileKey> = {
        let mut tiles = lock(&source.requests.tiles);
        if tiles.in_flight.contains(source.path) {
            return;
        }
        tiles.keep_only(source.path, keys);
        keys.iter()
            .copied()
            .filter(|&key| !tiles.holds(source.path, key))
            .collect()
    };
    if !missing.is_empty() {
        source
            .requests
            .decode_tiles(source.cache, source.path, missing, ctx);
    }
}

/// The tiles `keys` of `source`'s photograph that one render needs, handed
/// over together: every one of them, in the order of the keys, once a task
/// has answered it or the cache holds it, or none. While any is missing,
/// starts one decode on the rayon pool for the missing ones, unless one is
/// already running for the photograph, and takes nothing, so the tiles
/// answered before stay kept for the frame that finds them all. Never
/// decodes on this thread, at any budget. With no `ctx` it asks for nothing:
/// it starts no decode for the tiles missing and checks no failed file for a
/// change.
///
/// Unlike [`display_tiles`] it drops no kept tile it does not ask for, since
/// several renders of one panel read one photograph and ask for different
/// tiles; [`MAX_KEPT_TILES`] bounds what a photograph keeps instead.
fn display_tile_set(
    source: PhotographSource<'_>,
    keys: &[TileKey],
    ctx: Option<&egui::Context>,
) -> DisplayTileSet {
    if source.requests.failed_asking(source, ctx) {
        return DisplayTileSet::Unreadable;
    }
    let missing = {
        let mut kept = lock(&source.requests.tiles);
        let mut ready = Vec::with_capacity(keys.len());
        let mut missing = Vec::new();
        for &key in keys {
            if let Some(tile) = kept.kept(source.path, key) {
                ready.push(tile);
                continue;
            }
            match source.cache.peek_tile(source.path, key) {
                PeekedTile::Ready(tile) => ready.push(tile),
                PeekedTile::Unreadable => return DisplayTileSet::Unreadable,
                // Asked for by no task: the next frame's layout is
                // `Building`, which queues the photograph.
                PeekedTile::Building => return DisplayTileSet::Loading,
                PeekedTile::NotReady => missing.push(key),
            }
        }
        if missing.is_empty() {
            for &key in keys {
                kept.take(source.path, key);
            }
            return DisplayTileSet::Ready(ready);
        }
        missing
    };
    if let Some(ctx) = ctx {
        source
            .requests
            .decode_tiles(source.cache, source.path, missing, ctx);
    }
    DisplayTileSet::Loading
}

/// `render` of `source`'s photograph, run on the GUI thread once every
/// photograph tile it needs is there, on a photograph made of exactly those
/// tiles ([`TiledPhotograph::from_tiles`]), which holds them for the render:
/// a tile the cache evicts after the peek is still read, and the render
/// never decodes.
///
/// `needs` are the tiles the render needed when it was last tried, which
/// the caller keeps for the picture until it is drawn, empty at first. With
/// the coarsest level's first tile, which gives the photograph's channels
/// before anything the render needs is there, they are asked for together
/// with [`display_tile_set`]; once all are there `render` runs, and a
/// render refused with [`PhotographReadError::NeedsTiles`] for tiles not
/// among `needs` adds them and asks again, on this frame. So a picture's
/// first frames ask for the layout and then the first tile, the frame that
/// has it learns the tiles the render needs, and the frame that has those
/// renders.
pub(crate) fn read_peeked<T>(
    source: PhotographSource<'_>,
    needs: &mut Vec<TileKey>,
    ctx: &egui::Context,
    render: impl FnMut(&TiledPhotograph) -> Result<T, PhotographReadError>,
) -> PeekedRead<T> {
    read_tiles(source, needs, Some(ctx), render)
}

/// [`read_peeked`] on what is already there, asking for nothing: `render`
/// runs once the photograph's layout is known and every tile it needs is in
/// the cache or kept from a decode that answered it, and otherwise the read
/// is [`PeekedRead::Loading`] with no header read or decode started. For the
/// Track Table's rows while they scroll, which draw what is there and ask
/// for photographs only once the rows stop.
pub(crate) fn read_peeked_without_asking<T>(
    source: PhotographSource<'_>,
    needs: &mut Vec<TileKey>,
    render: impl FnMut(&TiledPhotograph) -> Result<T, PhotographReadError>,
) -> PeekedRead<T> {
    read_tiles(source, needs, None, render)
}

/// [`read_peeked`], which with no `ctx` asks for nothing.
fn read_tiles<T>(
    source: PhotographSource<'_>,
    needs: &mut Vec<TileKey>,
    ctx: Option<&egui::Context>,
    mut render: impl FnMut(&TiledPhotograph) -> Result<T, PhotographReadError>,
) -> PeekedRead<T> {
    let layout = match layout_of(source, ctx) {
        DisplayLayout::Known(layout) => layout,
        DisplayLayout::Loading => return PeekedRead::Loading,
        // The run of *Build mipmaps* that makes its mipmap ends with a
        // repaint.
        DisplayLayout::Building(_) => return PeekedRead::Building,
        DisplayLayout::Unreadable | DisplayLayout::WorkspaceFileError(_) => {
            return PeekedRead::Unreadable
        }
    };
    let Some(coarsest) = layout.levels().checked_sub(1) else {
        return PeekedRead::Unreadable;
    };
    let first = TileKey {
        level: coarsest as u8,
        col: 0,
        row: 0,
    };
    // Each pass either answers or adds to `needs`, which only the layout's
    // tiles can be named in, so the loop ends.
    loop {
        let mut keys = vec![first];
        keys.extend(needs.iter().copied().filter(|&key| key != first));
        let tiles = match display_tile_set(source, &keys, ctx) {
            DisplayTileSet::Ready(tiles) => tiles,
            DisplayTileSet::Loading => return PeekedRead::Loading,
            DisplayTileSet::Unreadable => return PeekedRead::Unreadable,
        };
        let photograph = match TiledPhotograph::from_tiles((*layout).clone(), tiles) {
            Ok(photograph) => photograph,
            Err(e) => return PeekedRead::Read(Err(e)),
        };
        match render(&photograph) {
            Err(PhotographReadError::NeedsTiles { keys })
                if keys.iter().any(|k| !needs.contains(k)) =>
            {
                for key in keys {
                    if !needs.contains(&key) {
                        needs.push(key);
                    }
                }
            }
            answer => return PeekedRead::Read(answer),
        }
    }
}

/// Level `level` of `path`, whose layout is `layout`, as a background task
/// reads it: its tiles asked for with [`PhotographCache::try_decode_tiles`],
/// which never waits, and the level built from the tiles that call answers,
/// at every budget, the tiles dropped once it is built. `None` when the
/// decode answered `Busy`.
fn read_level(
    cache: &PhotographCache,
    path: &Path,
    layout: &PyramidLayout,
    level: usize,
) -> Result<Option<Arc<ImageU8>>, TileError> {
    let (cols, rows) = layout.tiles(level);
    let keys: Vec<TileKey> = (0..rows)
        .flat_map(|row| {
            (0..cols).map(move |col| TileKey {
                level: level as u8,
                col,
                row,
            })
        })
        .collect();
    match cache.try_decode_tiles(path, &keys, &Progress::none())? {
        TryDecode::Busy => Ok(None),
        TryDecode::Decoded(tiles) => level_from_tiles(path, layout, level, &tiles).map(Some),
    }
}

/// Level `level` of `layout`'s photograph, at `path`, built from `tiles`, the
/// level's tiles with their borders as [`PhotographCache::try_decode_tiles`]
/// answers them: each tile's core copied to its place. A level the layout
/// does not have, or a tile whose place and size are not the ones the layout
/// gives it, which only a file that changed size since its layout was read
/// can give, is [`TileError::OutOfRange`].
fn level_from_tiles(
    path: &Path,
    layout: &PyramidLayout,
    level: usize,
    tiles: &[Arc<PhotographTile>],
) -> Result<Arc<ImageU8>, TileError> {
    let out_of_range = |why: String| TileError::OutOfRange {
        path: path.to_path_buf(),
        why,
    };
    if level >= layout.levels() {
        return Err(out_of_range(format!(
            "the layout has no level {level}, only {}",
            layout.levels()
        )));
    }
    let (width, height) = layout.level_size(level);
    let ts = layout.tile_size();
    let channels = tiles.first().map_or(3, |tile| tile.pixels.channels());
    let c = channels as usize;
    let row_len = width as usize * c;
    let mut data = vec![0; row_len * height as usize];
    for tile in tiles {
        // The core the layout gives the tile, so a file whose size changed
        // since the layout was read cannot leave a part of the level unset.
        let place = |index: u32, side: u32| {
            let origin = u64::from(index) * u64::from(ts);
            (
                origin,
                (u64::from(side).saturating_sub(origin)).min(u64::from(ts)),
            )
        };
        let (x0, w) = place(tile.key.col, width);
        let (y0, h) = place(tile.key.row, height);
        let fits = usize::from(tile.key.level) == level
            && [u64::from(tile.origin[0]), u64::from(tile.origin[1])] == [x0, y0]
            && [u64::from(tile.size[0]), u64::from(tile.size[1])] == [w, h]
            && w > 0
            && h > 0
            && tile.pixels.channels() == channels
            && u64::from(tile.pixels.width()) == w + 2
            && u64::from(tile.pixels.height()) == h + 2;
        if !fits {
            return Err(out_of_range(format!(
                "the tile at column {}, row {} of level {level} is not the one \
                 the layout's {width} x {height} gives",
                tile.key.col, tile.key.row
            )));
        }
        let [x0, y0] = tile.origin;
        let [w, h] = tile.size;
        let stride = tile.pixels.width() as usize * c;
        let span = w as usize * c;
        for y in 0..h as usize {
            // The core starts one row and one column inside the border.
            let from = (y + 1) * stride + c;
            let to = (y0 as usize + y) * row_len + x0 as usize * c;
            data[to..to + span].copy_from_slice(&tile.pixels.data()[from..from + span]);
        }
    }
    Ok(Arc::new(ImageU8::new(width, height, channels, data)))
}

/// Level `level` of `source`'s photograph, whose layout is `layout`, for the
/// 3D view's camera-view background, starting one task on the rayon pool that
/// reads it unless a decode is already running for the photograph. Never
/// decodes on this thread, at any budget.
///
/// The level a task answered is handed over once, as the background copies it
/// into its texture. Asking for another photograph or level drops it, since it
/// was answered for a background that has moved on, so at most one level is
/// kept.
pub(crate) fn display_level(
    source: PhotographSource<'_>,
    layout: &Arc<PyramidLayout>,
    level: usize,
    ctx: &egui::Context,
) -> DisplayLevel {
    if source.requests.failed(source, ctx) {
        return DisplayLevel::Unreadable;
    }
    let kept = lock(&source.requests.tiles).level.take();
    if let Some(kept) = kept {
        if kept.path == source.path && kept.level == level {
            return DisplayLevel::Ready(kept.pixels);
        }
    }
    source
        .requests
        .decode_level(source.cache, source.path, layout, level, ctx);
    DisplayLevel::Loading
}
