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
//!   level's tiles with [`PhotographCache::try_decode_tiles`] and then reads
//!   the level with [`PhotographCache::get_level`], or under a cache of
//!   budget 0 builds it from the tiles that call answers. It shares the tile
//!   decodes' limit of one task per photograph and their remembered
//!   failures, and keeps the one level it last answered until the background
//!   asks for it.
//! - Track View and the Track Table still take the whole pyramid from
//!   [`display_photograph`] ([`PhotographCache::peek_state`], and
//!   [`PhotographCache::get`] on the rayon pool on a miss).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use sfmtool_core::camera::image::{ImageU8, ImageU8Pyramid};
use sfmtool_core::camera::{
    PeekedLayout, PeekedPhotograph, PeekedTile, PhotographCache, PhotographTile, PyramidLayout,
    TileError, TileKey, TryDecode,
};
use sfmtool_core::progress::Progress;
use sfmtool_core::SfmrReconstruction;

use super::photograph_path;

#[cfg(test)]
mod tests;

/// What a panel has to draw a photograph with on this frame.
#[derive(Clone)]
pub(crate) enum DisplayPhotograph {
    /// Decoded; level 0 is the photograph.
    Decoded(Arc<ImageU8Pyramid>),
    /// Being decoded on a worker, or about to be. A repaint follows the
    /// decode.
    Decoding,
    /// The file could not be read or decoded, or the image index is not in the image
    /// set.
    Unreadable,
}

/// What a panel knows of a photograph's layout on this frame.
#[derive(Clone, Debug)]
pub(crate) enum DisplayLayout {
    /// Its level sizes and tile grid: its tiles may be asked for.
    Known(Arc<PyramidLayout>),
    /// Its header is being read on a worker, or about to be. A repaint
    /// follows the read.
    Loading,
    /// The file could not be read or decoded.
    Unreadable,
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
/// what one decode answered, which a panel asks for at most 16 of
/// (`image_detail::photograph_tiles`'s upload cap), since every ask for the
/// photograph drops the kept tiles it does not ask for: at most 64 tiles of
/// 514 x 514 RGB, about 51 MB, in all.
const MAX_ANSWERED_PHOTOGRAPHS: usize = 4;

/// A file's length and modification time, which tell a file that changed
/// since a decode of it failed: `None` when it cannot be read.
type FileStamp = Option<(u64, Option<std::time::SystemTime>)>;

/// `path`'s [`FileStamp`] now.
fn file_stamp(path: &Path) -> FileStamp {
    std::fs::metadata(path)
        .ok()
        .map(|m| (m.len(), m.modified().ok()))
}

/// A task to run off the GUI thread.
type Task = Box<dyn FnOnce() + Send>;

/// Where tasks are started: the rayon pool, or in the tests a queue the test
/// runs by hand.
type Spawner = Arc<dyn Fn(Task) + Send + Sync>;

/// What a photograph's last decode task answered, kept for the panels.
enum Answered {
    /// The tiles no panel has taken yet.
    Tiles(HashMap<TileKey, Arc<PhotographTile>>),
    /// The photograph could not be read or decoded, when the file had this
    /// stamp. A file whose stamp has changed since is asked for again.
    Failed(FileStamp),
}

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
    /// What the tile decodes answered, and every decode's failure, oldest
    /// first, at most [`MAX_ANSWERED_PHOTOGRAPHS`] paths.
    answered: Vec<(PathBuf, Answered)>,
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

    /// The stamp `path`'s file had when its decode failed, if it failed.
    fn failed(&mut self, path: &Path) -> Option<FileStamp> {
        match self.answered(path) {
            Some(Answered::Failed(stamp)) => Some(*stamp),
            _ => None,
        }
    }

    /// Keep what a decode of `path` that failed with `e` tells later frames.
    /// A cancelled decode tells nothing, and the next frame asks again. An
    /// unreadable photograph is remembered with its file's stamp. A key or a
    /// level the photograph does not have comes from a layout read before the
    /// file changed size: the cache forgets the path, so the next frame reads
    /// the header again rather than ask for the same pixels on every frame.
    fn failed_with(&mut self, cache: &PhotographCache, path: &Path, e: TileError) {
        match e {
            TileError::Cancelled => {}
            TileError::Unreadable { .. } => {
                self.keep(path.to_path_buf(), Answered::Failed(file_stamp(path)));
            }
            TileError::OutOfRange { .. } => cache.forget(path),
        }
    }

    /// Drop `path`'s answer.
    fn forget(&mut self, path: &Path) {
        self.answered.retain(|(p, _)| p != path);
    }

    /// Drop `path`'s kept tiles that are not among `keys`, the ones a panel
    /// asks for now: tiles a decode answered for a view that has moved on.
    fn keep_only(&mut self, path: &Path, keys: &[TileKey]) {
        let Some(index) = self.answered.iter().position(|(p, _)| p == path) else {
            return;
        };
        if let Answered::Tiles(tiles) = &mut self.answered[index].1 {
            tiles.retain(|key, _| keys.contains(key));
            if tiles.is_empty() {
                self.answered.remove(index);
            }
        }
    }

    /// How many answered tiles are kept, of every photograph.
    #[cfg(test)]
    fn kept_tiles(&self) -> usize {
        self.answered
            .iter()
            .map(|(_, answered)| match answered {
                Answered::Tiles(tiles) => tiles.len(),
                Answered::Failed(_) => 0,
            })
            .sum()
    }

    /// Whether `key` of `path` has been answered and not yet taken.
    fn holds(&mut self, path: &Path, key: TileKey) -> bool {
        matches!(self.answered(path), Some(Answered::Tiles(tiles)) if tiles.contains_key(&key))
    }

    /// Take `key` of `path` if a task answered it.
    fn take(&mut self, path: &Path, key: TileKey) -> Option<Arc<PhotographTile>> {
        let index = self.answered.iter().position(|(p, _)| p == path)?;
        let Answered::Tiles(tiles) = &mut self.answered[index].1 else {
            return None;
        };
        let tile = tiles.remove(&key);
        if tiles.is_empty() {
            self.answered.remove(index);
        }
        tile
    }

    /// Keep what a task answered for `path` as its newest answer, dropping
    /// the oldest past [`MAX_ANSWERED_PHOTOGRAPHS`].
    fn keep(&mut self, path: PathBuf, answer: Answered) {
        let answer = match (self.answered.iter().position(|(p, _)| *p == path), answer) {
            (Some(index), Answered::Tiles(more)) => match self.answered.remove(index).1 {
                Answered::Tiles(mut tiles) => {
                    tiles.extend(more);
                    Answered::Tiles(tiles)
                }
                Answered::Failed(_) => Answered::Tiles(more),
            },
            (Some(index), failed @ Answered::Failed(_)) => {
                self.answered.remove(index);
                failed
            }
            (None, answer) => answer,
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
    /// The paths a [`display_photograph`] decode is running for.
    in_flight: Arc<Mutex<HashSet<PathBuf>>>,
    /// The paths a header read is running for.
    layouts_in_flight: Arc<Mutex<HashSet<PathBuf>>>,
    tiles: Arc<Mutex<TileRequests>>,
    spawn: Spawner,
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
            in_flight: Arc::default(),
            layouts_in_flight: Arc::default(),
            tiles: Arc::default(),
            spawn,
        }
    }

    /// Start decoding `path` into `cache` on the rayon pool unless a decode of
    /// it is already running, and repaint `ctx` when it ends.
    fn request(&self, cache: &Arc<PhotographCache>, path: &Path, ctx: &egui::Context) {
        if !lock(&self.in_flight).insert(path.to_path_buf()) {
            return;
        }
        let cache = Arc::clone(cache);
        let in_flight = Arc::clone(&self.in_flight);
        let path = path.to_path_buf();
        let ctx = ctx.clone();
        (self.spawn)(Box::new(move || {
            let _ = cache.get(&path, &Progress::none());
            // After the get, so a frame that no longer finds the path in flight
            // finds its entry in the cache.
            lock(&in_flight).remove(&path);
            ctx.request_repaint();
        }));
    }

    /// Start reading `path`'s header into `cache` unless a read of it is
    /// already running, and repaint `ctx` when it ends.
    fn request_layout(&self, cache: &Arc<PhotographCache>, path: &Path, ctx: &egui::Context) {
        if !lock(&self.layouts_in_flight).insert(path.to_path_buf()) {
            return;
        }
        let cache = Arc::clone(cache);
        let in_flight = Arc::clone(&self.layouts_in_flight);
        let path = path.to_path_buf();
        let ctx = ctx.clone();
        (self.spawn)(Box::new(move || {
            cache.request_layout(&path);
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
            {
                // One lock for the answer and the end of the flight, so a
                // frame that finds the path no longer in flight finds its
                // answer.
                let mut tiles = lock(&tiles);
                match answer {
                    Ok(TryDecode::Decoded(found)) => {
                        let found = found.into_iter().map(|tile| (tile.key, tile)).collect();
                        tiles.keep(path.clone(), Answered::Tiles(found));
                    }
                    // Nothing was decoded, and the next frame asks again.
                    Ok(TryDecode::Busy) => {}
                    Err(e) => tiles.failed_with(&cache, &path, e),
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
    /// the level's tiles with [`PhotographCache::try_decode_tiles`], then
    /// reads the level with [`PhotographCache::get_level`], or under a cache
    /// of budget 0, which keeps nothing for `get_level` to find, builds it
    /// from the tiles that call answers. The level is kept for
    /// [`display_level`], and `ctx` repainted when the task ends: at once, or
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
                    Err(e) => tiles.failed_with(&cache, &path, e),
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

    /// Answer whether `path`'s decode failed, and if it did, start one check
    /// on the rayon pool of whether the file has changed since, which drops
    /// the failure and repaints when it has, so a file that is fixed loads
    /// again. The check stats the file, off this thread.
    fn failed(&self, path: &Path, ctx: &egui::Context) -> bool {
        let stamp = {
            let mut tiles = lock(&self.tiles);
            let Some(stamp) = tiles.failed(path) else {
                return false;
            };
            if !tiles.in_flight.insert(path.to_path_buf()) {
                return true;
            }
            stamp
        };
        let tiles = Arc::clone(&self.tiles);
        let path = path.to_path_buf();
        let ctx = ctx.clone();
        (self.spawn)(Box::new(move || {
            let changed = file_stamp(&path) != stamp;
            {
                let mut tiles = lock(&tiles);
                if changed {
                    tiles.forget(&path);
                }
                tiles.in_flight.remove(&path);
            }
            if changed {
                ctx.request_repaint();
            }
        }));
        true
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

    /// Whether a decode of `path` is running.
    #[cfg(test)]
    pub(crate) fn is_in_flight(&self, path: &Path) -> bool {
        lock(&self.in_flight).contains(path)
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

/// The photograph of image `index` of `recon` as `cache` holds it, asking
/// `requests` to decode it in the background when it holds nothing yet.
///
/// Never reads the file on this thread, except under a cache with a budget of
/// 0: that cache keeps nothing, so a background decode would never be seen
/// and would be asked for again every frame, and the photograph is decoded
/// here instead, as it was before panels stopped decoding.
///
/// A remembered failure is answered from the cache without a stat of the
/// file, so a photograph that appears after it failed shows once something
/// else reads it with `get` or the entry is forgotten.
pub(crate) fn display_photograph(
    cache: &Arc<PhotographCache>,
    requests: &PhotographRequests,
    recon: &SfmrReconstruction,
    index: usize,
    ctx: &egui::Context,
) -> DisplayPhotograph {
    let Some(path) = photograph_path(recon, index) else {
        return DisplayPhotograph::Unreadable;
    };
    match cache.peek_state(&path) {
        PeekedPhotograph::Decoded(pyramid) => DisplayPhotograph::Decoded(pyramid),
        PeekedPhotograph::Unreadable => DisplayPhotograph::Unreadable,
        PeekedPhotograph::NotDecoded if cache.budget_bytes() == 0 => {
            match cache.get(&path, &Progress::none()) {
                Ok(pyramid) => DisplayPhotograph::Decoded(pyramid),
                Err(_) => DisplayPhotograph::Unreadable,
            }
        }
        PeekedPhotograph::NotDecoded => {
            requests.request(cache, &path, ctx);
            DisplayPhotograph::Decoding
        }
    }
}

/// The layout of `source`'s photograph, starting one read of its header on
/// the rayon pool when it is not known. Reads nothing on this thread, at any
/// budget, and allocates nothing for a layout the cache already knows.
///
/// A photograph whose last decode task failed is answered as unreadable,
/// which is what a cache of budget 0, which remembers no failure, would
/// otherwise never say, until its file changes.
pub(crate) fn display_layout(source: PhotographSource<'_>, ctx: &egui::Context) -> DisplayLayout {
    if source.requests.failed(source.path, ctx) {
        return DisplayLayout::Unreadable;
    }
    match source.cache.layout(source.path) {
        PeekedLayout::Known(layout) => DisplayLayout::Known(layout),
        PeekedLayout::Unreadable(_) => DisplayLayout::Unreadable,
        PeekedLayout::NotKnown => {
            source
                .requests
                .request_layout(source.cache, source.path, ctx);
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
    if source.requests.failed(source.path, ctx) {
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
    if source.requests.failed(source.path, ctx) {
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

/// Level `level` of `path`, whose layout is `layout`, as a background task
/// reads it: its tiles asked for with [`PhotographCache::try_decode_tiles`],
/// then the level read with [`PhotographCache::get_level`], or under a cache
/// of budget 0 built from the tiles that call answers. `None` when the
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
    let progress = Progress::none();
    let tiles = match cache.try_decode_tiles(path, &keys, &progress)? {
        TryDecode::Busy => return Ok(None),
        TryDecode::Decoded(tiles) => tiles,
    };
    let pixels = if cache.budget_bytes() == 0 {
        level_from_tiles(path, layout, level, &tiles)?
    } else {
        // A hit: the decode has just made the photograph, or found it held.
        // The tiles are not needed for it, so they are not held meanwhile.
        drop(tiles);
        cache.get_level(path, level, &progress)?
    };
    check_level_size(path, layout, level, pixels).map(Some)
}

/// Level `level` of `layout`'s photograph, at `path`, built from `tiles`, the
/// level's tiles with their borders, as a cache of budget 0 answers them:
/// each tile's core copied to its place. A level the layout does not have, or
/// a tile that does not fit inside the level, which only a file that changed
/// size since its layout was read can give, is [`TileError::OutOfRange`].
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
    let channels = tiles.first().map_or(3, |tile| tile.pixels.channels());
    let c = channels as usize;
    let row_len = width as usize * c;
    let mut data = vec![0; row_len * height as usize];
    for tile in tiles {
        let [x0, y0] = tile.origin;
        let [w, h] = tile.size;
        let fits = u64::from(x0) + u64::from(w) <= u64::from(width)
            && u64::from(y0) + u64::from(h) <= u64::from(height)
            && tile.pixels.channels() == channels
            && tile.pixels.width() == w + 2
            && tile.pixels.height() == h + 2;
        if !fits {
            return Err(out_of_range(format!(
                "the tile at column {}, row {} of level {level} does not fit its \
                 {width} x {height}",
                tile.key.col, tile.key.row
            )));
        }
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

/// `pixels`, level `level` of the photograph at `path`, once it is known to be
/// the size `layout` gives that level: a file that changed size since its
/// layout was read gives another, which is [`TileError::OutOfRange`].
fn check_level_size(
    path: &Path,
    layout: &PyramidLayout,
    level: usize,
    pixels: Arc<ImageU8>,
) -> Result<Arc<ImageU8>, TileError> {
    let (width, height) = layout.level_size(level);
    if (pixels.width(), pixels.height()) == (width, height) {
        return Ok(pixels);
    }
    Err(TileError::OutOfRange {
        path: path.to_path_buf(),
        why: format!(
            "level {level} is {} x {}, not the layout's {width} x {height}",
            pixels.width(),
            pixels.height()
        ),
    })
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
    if source.requests.failed(source.path, ctx) {
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
