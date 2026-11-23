// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Decoded photographs as photograph tiles, keyed by file path, shared across
//! threads and bounded by a byte budget, with the whole decodes in progress
//! bounded by a build budget.
//!
//! Decoding a photograph and building the pyramid the patch samplers read is
//! the expensive part of every operation that reads a reconstruction's pixels.
//! A [`PhotographCache`] cuts each decode's pyramid into photograph tiles,
//! keeps the tiles and drops the pyramid, so that whichever thread asks next
//! for the same file gets the same pixels without reading it again. When the
//! bytes it holds pass its budget it drops tiles, those of a decode that no
//! request asked for first, then the least recently used.
//!
//! Its callers hold it in an `Arc` and call it through `&self`: a worker thread
//! puts what it decodes where the next worker and the GUI thread find it.
//! [`PhotographCache::get`] answers a [`TiledPhotograph`] that holds no tile
//! and reads each through the cache, decoding the photograph only when none
//! of its tiles is kept, and again for a read of a tile the cache evicted
//! since; it waits while another thread decodes the same path, and stops
//! waiting when its caller is cancelled. [`PhotographCache::peek`] never reads a file and
//! never waits, which is what a caller that draws every frame wants. Every
//! whole decode is reserved on the cache's build count first
//! ([`PhotographCache::reserve_builds`]), so the decodes in progress on one
//! cache stay within its build budget. A panel asks for a photograph's layout
//! and tiles with [`PhotographCache::layout`], [`PhotographCache::peek_tile`]
//! and [`PhotographCache::try_decode_tiles`], which never waits. The design is
//! in the spec `specs/core/camera/photograph-cache.md`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, SystemTime};

use rayon::prelude::*;

use crate::camera::image::{image_dimensions, ImageU8};
use crate::camera::mipmap::{PyramidLayout, MAX_JPEG_SIDE, PHOTOGRAPH_TILE_SIZE};
use crate::camera::tiled_photograph::{
    cut_image, tile_count, CutImage, PhotographReadError, PhotographTile, TileKey, TiledPhotograph,
};
use crate::progress::{Cancelled, Progress};

mod tiles;

pub use tiles::{PeekedLayout, PeekedTile, TileError, TryDecode};

/// How long a caller waiting for another caller's decode of the same path
/// waits on the slot before it looks at its own `Progress` for cancellation
/// again: a cancelled waiter stops within about this long.
pub const SLOT_WAIT_POLL: Duration = Duration::from_millis(50);

/// The environment variable that overrides [`default_budget_bytes`], in
/// megabytes (10^6 bytes), written as a decimal integer.
pub const BUDGET_ENV_VAR: &str = "SFM_EXPLORER_PHOTOGRAPH_CACHE_MB";

/// The environment variable that overrides [`default_build_budget_bytes`], in
/// megabytes (10^6 bytes), written as a decimal integer.
pub const BUILD_BUDGET_ENV_VAR: &str = "SFM_EXPLORER_BUILD_BUDGET_MB";

const GIB: u64 = 1 << 30;

/// The byte budget a cache gets by default: a quarter of physical memory,
/// clamped to between 1 GiB and 16 GiB, or 4 GiB where physical memory cannot
/// be read.
///
/// The environment variable [`BUDGET_ENV_VAR`] overrides it with a number of
/// megabytes (10^6 bytes); `0` there gives a cache that keeps nothing. A value
/// that does not parse as a whole number is ignored.
///
/// Physical memory is read by [`crate::machine::physical_memory_bytes`], and a
/// target where it cannot be read takes the 4 GiB fallback.
pub fn default_budget_bytes() -> u64 {
    let env = std::env::var(BUDGET_ENV_VAR).ok();
    budget_from(env.as_deref(), crate::machine::physical_memory_bytes())
}

/// [`default_budget_bytes`] and [`default_build_budget_bytes`] with their two
/// inputs passed in, so they can be tested without touching the environment.
fn budget_from(env_megabytes: Option<&str>, physical_bytes: Option<u64>) -> u64 {
    if let Some(mb) = env_megabytes.and_then(|v| v.trim().parse::<u64>().ok()) {
        return mb.saturating_mul(1_000_000);
    }
    match physical_bytes {
        Some(bytes) => (bytes / 4).clamp(GIB, 16 * GIB),
        None => 4 * GIB,
    }
}

/// The build budget a cache gets by default, worked out as
/// [`default_budget_bytes`] is: a quarter of physical memory, clamped to
/// between 1 GiB and 16 GiB, or 4 GiB where physical memory cannot be read.
/// It keeps no state.
///
/// The environment variable [`BUILD_BUDGET_ENV_VAR`] overrides it with a
/// number of megabytes (10^6 bytes). `0` there lets each caller decode one
/// photograph at a time, since [`PhotographCache::reserve_builds`] grants one
/// decode when none fits, and lets panels decode one photograph at a time,
/// since [`PhotographCache::try_reserve_build`] grants a decode that does not
/// fit when nothing is reserved. A value that does not parse as a whole number
/// is ignored.
pub fn default_build_budget_bytes() -> u64 {
    let env = std::env::var(BUILD_BUDGET_ENV_VAR).ok();
    budget_from(env.as_deref(), crate::machine::physical_memory_bytes())
}

/// The bytes a whole decode of a photograph is estimated to hold while it
/// runs, per pixel of the photograph: its RGB pixels (3) and the tiles cut
/// from them, which grow to about as many bytes again while level 0 is cut
/// and the image is still held. Measured, a decode with the decoder's own
/// buffers peaks at about 6.4 bytes per pixel.
const WHOLE_DECODE_BYTES_PER_PIXEL: u64 = 5;

/// The estimated bytes of a whole decode of a `width x height` photograph,
/// what its decode reserves on the build count. Saturates, since `get_many`
/// sizes a miss from a header it has not yet checked against
/// [`MAX_JPEG_SIDE`].
fn whole_decode_bytes(width: u32, height: u32) -> u64 {
    (u64::from(width) * u64::from(height)).saturating_mul(WHOLE_DECODE_BYTES_PER_PIXEL)
}

/// The settings a [`PhotographCache`] is made with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhotographCacheOptions {
    /// Bytes of photograph tiles kept before tiles are dropped, the unused
    /// first and then the least recently used; 0 keeps nothing. By default
    /// [`default_budget_bytes`].
    pub budget_bytes: u64,
    /// Estimated bytes of the whole decodes in progress on this cache at
    /// once, what [`PhotographCache::reserve_builds`] and
    /// [`PhotographCache::try_reserve_build`] reserve against. By default
    /// [`default_build_budget_bytes`].
    pub build_budget_bytes: u64,
    /// The side of the photograph tiles the cache cuts its decodes into, by
    /// default [`PHOTOGRAPH_TILE_SIZE`]; only tests change it.
    pub tile_size: u32,
}

impl Default for PhotographCacheOptions {
    /// [`default_budget_bytes`], [`default_build_budget_bytes`] and
    /// [`PHOTOGRAPH_TILE_SIZE`].
    fn default() -> Self {
        Self {
            budget_bytes: default_budget_bytes(),
            build_budget_bytes: default_build_budget_bytes(),
            tile_size: PHOTOGRAPH_TILE_SIZE,
        }
    }
}

/// Whole decodes reserved on one cache's build count, each of the same
/// estimated bytes. [`Self::release_one`] gives one decode's bytes back as it
/// ends, and dropping the reservation gives back the rest, so a caller that
/// stops early, a panic included, leaves nothing reserved. Made by
/// [`PhotographCache::reserve_builds`] and
/// [`PhotographCache::try_reserve_build`].
#[derive(Debug)]
pub struct BuildReservation<'a> {
    count: &'a AtomicU64,
    bytes_per_build: u64,
    granted: usize,
    released: AtomicU64,
}

impl BuildReservation<'_> {
    /// How many decodes were granted.
    pub fn granted(&self) -> usize {
        self.granted
    }

    /// Give back one granted decode's bytes, as that decode ends. Does nothing
    /// once every granted decode has been given back.
    pub fn release_one(&self) {
        let granted = self.granted as u64;
        let took = update(&self.released, |released| {
            (released < granted).then_some(released + 1)
        });
        if took.is_some() {
            give_back(self.count, self.bytes_per_build);
        }
    }
}

impl Drop for BuildReservation<'_> {
    fn drop(&mut self) {
        let granted = self.granted as u64;
        let released = self.released.get_mut();
        let left = granted.saturating_sub(*released);
        *released = granted;
        give_back(self.count, self.bytes_per_build.saturating_mul(left));
    }
}

/// Set `atomic` to `next(value)` with a compare-exchange loop, unless `next`
/// answers `None`; the value it replaced, or `None` when it changed nothing.
fn update(atomic: &AtomicU64, next: impl Fn(u64) -> Option<u64>) -> Option<u64> {
    let mut value = atomic.load(Ordering::Acquire);
    loop {
        let new = next(value)?;
        match atomic.compare_exchange_weak(value, new, Ordering::AcqRel, Ordering::Acquire) {
            Ok(old) => return Some(old),
            Err(now) => value = now,
        }
    }
}

/// Take `bytes` off a build count, stopping at 0. Only a reservation whose
/// bytes saturated the count could otherwise take it below 0.
fn give_back(count: &AtomicU64, bytes: u64) {
    if bytes != 0 {
        update(count, |n| Some(n.saturating_sub(bytes)));
    }
}

/// The build count's share for one caller's decode, given back when it is
/// dropped or released.
enum Share<'r> {
    /// Nothing reserved: a `get` reserves its one decode once it has read the
    /// photograph's size from its header.
    None,
    /// A `get`'s own reservation of its one decode.
    Own(BuildReservation<'r>),
    /// One decode of a `get_many` group's reservation, reserved when the
    /// group was decided.
    Group(GroupShare<'r>),
}

impl Share<'_> {
    /// Give the share back, at most once.
    fn release(&mut self) {
        match std::mem::replace(self, Share::None) {
            Share::None => {}
            Share::Own(reservation) => drop(reservation),
            Share::Group(share) => drop(share),
        }
    }
}

/// One member's decode of a `get_many` group's reservation, given back when
/// the member ends, a panic included.
struct GroupShare<'r>(&'r BuildReservation<'r>);

impl Drop for GroupShare<'_> {
    fn drop(&mut self) {
        self.0.release_one();
    }
}

/// Why [`PhotographCache::get`] has no photograph for a path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GetError {
    /// The file does not exist.
    NotFound(PathBuf),
    /// The file could not be read or decoded; the message names it.
    Unreadable(String),
    /// A side of 65536 px or more (over [`MAX_JPEG_SIDE`]), read from the
    /// header before any decode. Every photograph with such a side is
    /// refused, whatever its workspace, since no JPEG frame could hold its
    /// full-resolution mip level.
    TooLarge {
        path: PathBuf,
        width: u32,
        height: u32,
    },
    /// The caller's `Progress` was cancelled while it decoded the photograph
    /// or waited for another caller's decode of it. Never remembered: the
    /// next `get` decodes or waits again.
    Cancelled,
}

impl std::fmt::Display for GetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(path) => write!(f, "{}: no such file", path.display()),
            Self::Unreadable(why) => write!(f, "{why}"),
            Self::TooLarge {
                path,
                width,
                height,
            } => write!(
                f,
                "{} is {width} x {height}, and a photograph with a side over \
                 {MAX_JPEG_SIDE} px cannot be read",
                path.display()
            ),
            Self::Cancelled => write!(f, "{Cancelled}"),
        }
    }
}

impl std::error::Error for GetError {}

/// What [`PhotographCache::stats`] reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhotographCacheStats {
    /// Photograph tiles kept, of every photograph.
    pub tiles: usize,
    /// Pixel bytes of the tiles kept, each tile with its border.
    pub bytes: u64,
    /// The budget the cache was made with.
    pub budget_bytes: u64,
    /// Answers that decoded nothing: every tile asked for was kept (for
    /// `get`, a tile of the photograph), the file is remembered as
    /// unreadable, or another caller's decode of the same path was waited on.
    /// Each path of a `get_many` counts as one `get`. The peeks are not
    /// counted, nor is a read of a photograph `get` answered that finds its
    /// tiles kept, nor a call that stopped waiting because it was cancelled.
    pub hits: u64,
    /// Decodes started, whether they succeeded, failed or were cancelled: a
    /// photograph none of whose tiles were kept, or one a tile of which was
    /// asked for or read and not kept, decoded again.
    pub misses: u64,
    /// Photographs decoded whole: every decode that read a photograph's
    /// pixels, including one whose caller was cancelled before its tiles
    /// were cut. A file whose header cannot be read, or that is refused as
    /// too large, is a miss but not a decode.
    pub decodes: u64,
    /// The build budget the cache was made with.
    pub build_budget_bytes: u64,
    /// The estimated bytes reserved on the build count now, for whole decodes
    /// in progress or granted to a `get_many` group.
    pub build_reserved_bytes: u64,
}

/// How a [`PhotographCache::get_many`] call's photographs were found. The three
/// counts add up to the number of paths asked for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GetManyTally {
    /// Decoded by this call.
    pub read: usize,
    /// Every tile already in the cache, or decoded by another thread while
    /// this call waited for it.
    pub reused: usize,
    /// Could not be read or decoded, now or when last tried.
    pub unreadable: usize,
}

/// What [`PhotographCache::peek_state`] found for a path: [`PhotographCache::peek`]
/// with a failure told apart from a photograph not yet decoded.
#[derive(Clone)]
pub enum PeekedPhotograph {
    /// The decoded photograph, every tile of which the cache keeps.
    Decoded(Arc<TiledPhotograph>),
    /// A decode was tried and the file could not be read or decoded. The
    /// answer `get` gives until the entry is forgotten or the file changes.
    Unreadable,
    /// No answer yet: the path was never asked for, a tile of it was
    /// evicted, or its decode is still running.
    NotDecoded,
}

/// A file's length and modification time, as a slot records them when it is
/// made. `None` when the file cannot be stat'ed.
type FileStamp = Option<(u64, Option<SystemTime>)>;

fn file_stamp(path: &Path) -> FileStamp {
    let meta = std::fs::metadata(path).ok()?;
    Some((meta.len(), meta.modified().ok()))
}

/// The error a failed read of `path` is answered and remembered as.
fn unreadable(path: &Path, e: ::image::ImageError) -> GetError {
    match e {
        ::image::ImageError::IoError(io) if io.kind() == std::io::ErrorKind::NotFound => {
            GetError::NotFound(path.to_path_buf())
        }
        e => GetError::Unreadable(format!("{}: {e}", path.display())),
    }
}

/// The photograph's size from its header, refused as [`GetError::TooLarge`]
/// with a side over [`MAX_JPEG_SIDE`]: the header decides the refusal, so a
/// photograph too large for the pyramid readers is never decoded.
fn header(path: &Path) -> Result<(u32, u32), GetError> {
    let (width, height) = image_dimensions(path).map_err(|e| unreadable(path, e))?;
    if width > MAX_JPEG_SIDE || height > MAX_JPEG_SIDE {
        return Err(GetError::TooLarge {
            path: path.to_path_buf(),
            width,
            height,
        });
    }
    Ok((width, height))
}

/// A tile the cache keeps, with what its eviction order reads.
struct Kept {
    tile: Arc<PhotographTile>,
    /// A request asked for the tile or read it: `try_decode_tiles`' keys,
    /// `get_tile`'s key and `peek_tile`'s. An unused tile is one only cut
    /// with the rest of its decode, and is evicted before every used one.
    used: bool,
    /// The cache's tick when the tile was kept or last asked for.
    tick: u64,
}

/// What a request asks a slot for: which tiles it is answered, and which its
/// decode keeps as used.
#[derive(Clone, Copy)]
enum Asked<'a> {
    /// The photograph read through the cache, `get`'s and `get_many`'s:
    /// answered no tile, once the photograph is made and a tile of it is
    /// kept, and a decode keeps every tile as unused. A decode is made only
    /// when no tile is kept; a read of a tile not kept decodes again.
    Photograph,
    /// Every tile of the photograph, `peek`'s and `insert`'s: answered every
    /// tile, and a decode keeps them all as unused.
    Every,
    /// The tiles `keys` that the layout has, `get_tile`'s and a panel's
    /// decode's: answered those, and a decode keeps them as used and the
    /// rest as unused.
    Keys(&'a [TileKey]),
}

/// A request's answer: the photograph's layout and the tiles it asked for,
/// every tile for [`Asked::Every`].
struct Got {
    layout: Arc<PyramidLayout>,
    tiles: Vec<Arc<PhotographTile>>,
    /// The channels of the photograph's tiles.
    channels: u32,
    /// For [`Asked::Photograph`], the slot the photograph reads its tiles
    /// from; `None` when the cache keeps none of them (a budget of 0, or a
    /// slot that left the map while its decode ran), and `tiles` is then
    /// every tile of the decode.
    slot: Option<Arc<Slot>>,
}

impl Got {
    /// The photograph the request is answered: one that reads its tiles
    /// through `cache` from the slot, or one made of the tiles, which for
    /// [`Asked::Every`], and for [`Asked::Photograph`] with no slot, are
    /// every tile of its layout.
    fn photograph(
        self,
        cache: &PhotographCache,
        path: &Path,
    ) -> Result<Arc<TiledPhotograph>, GetError> {
        if let Some(slot) = self.slot {
            let layout = (*self.layout).clone();
            let through = CachedTiles {
                cache: cache.handle(),
                path: path.to_path_buf(),
                slot,
                layout: self.layout,
            };
            return Ok(Arc::new(TiledPhotograph::read_through(
                layout,
                self.channels,
                through,
            )));
        }
        TiledPhotograph::from_tiles((*self.layout).clone(), self.tiles)
            .map(Arc::new)
            // Every layout has a tile, and the tiles are the layout's own, cut
            // from one image; this answers a broken invariant rather than
            // panic on it.
            .map_err(|e| GetError::Unreadable(e.to_string()))
    }
}

/// What a photograph [`PhotographCache::get`] or [`PhotographCache::get_many`]
/// answers reads its tiles through: the cache, the path, the slot it was
/// answered from and the layout of the decode that made it.
///
/// A read takes the tiles the slot keeps; when one is not kept it reads the
/// file again as [`PhotographCache::get_tile`] does, decoding the photograph,
/// reserving with [`PhotographCache::reserve_builds`] and waiting on the slot
/// while another caller decodes it, polling the reader's `Progress`. A file
/// whose length or modification time is not the slot's any more, or whose
/// layout is not the one recorded, fails the read with
/// [`PhotographReadError::Changed`], so a kernel never reads two versions of
/// one photograph.
pub(crate) struct CachedTiles {
    cache: PhotographCache,
    path: PathBuf,
    slot: Arc<Slot>,
    layout: Arc<PyramidLayout>,
}

impl CachedTiles {
    /// The tiles `keys`, in order, each a tile of the layout: from the slot
    /// when it keeps every one of them, marking them used, and otherwise
    /// from one decode of the file that keeps them as used.
    pub(crate) fn read(
        &self,
        keys: &[TileKey],
        progress: &Progress<'_>,
    ) -> Result<Vec<Arc<PhotographTile>>, PhotographReadError> {
        if let Some(&key) = keys.iter().find(|&&key| !has_key(&self.layout, key)) {
            return Err(PhotographReadError::OutOfRange {
                path: Some(self.path.clone()),
                why: format!(
                    "level {} has no tile at column {}, row {}",
                    key.level, key.col, key.row
                ),
            });
        }
        let cache = &self.cache;
        {
            let tick = cache.next_tick();
            let mut data = self.slot.lock();
            if matches!(&data.made, Some(Ok(layout)) if **layout == *self.layout) {
                if let Some(got) = data.kept(Asked::Keys(keys), tick, &self.slot) {
                    return Ok(got.tiles);
                }
            }
        }
        // A tile is not kept: read the file again, as it was when the slot
        // was made.
        progress.check_cancel()?;
        let stamp = file_stamp(&self.path);
        if stamp.is_some() && stamp != self.slot.stamp {
            return Err(PhotographReadError::Changed {
                path: self.path.clone(),
            });
        }
        let (value, _) = cache.fetch(&self.path, progress, Share::None, Asked::Keys(keys));
        let got = value.map_err(|e| read_error(&self.path, e))?;
        if *got.layout != *self.layout || got.tiles.len() != keys.len() {
            return Err(PhotographReadError::Changed {
                path: self.path.clone(),
            });
        }
        Ok(got.tiles)
    }
}

/// The [`PhotographReadError`] of a failed read of `path`'s file.
fn read_error(path: &Path, e: GetError) -> PhotographReadError {
    match e {
        GetError::Unreadable(why) => PhotographReadError::Unreadable {
            path: Some(path.to_path_buf()),
            why,
        },
        e => e.into(),
    }
}

/// A slot's state, behind its mutex.
#[derive(Default)]
struct SlotData {
    /// One caller is decoding the photograph into the slot; the others wait
    /// on [`Slot::ready`]. The tiles kept from an earlier decode stay
    /// readable while it runs.
    making: bool,
    /// The photograph's layout once a decode made it, or the failure
    /// remembered for the file; `None` before any decode finished.
    made: Option<Result<Arc<PyramidLayout>, GetError>>,
    /// The tiles kept, by key. Empty unless the slot is in the map: a slot
    /// leaves the map with no tile.
    tiles: HashMap<TileKey, Kept>,
    /// Bytes of `tiles`' pixels, what the slot adds to [`Inner::bytes`].
    bytes: u64,
    /// The channels of the tiles of the decode that made the slot.
    channels: u32,
}

impl SlotData {
    /// The tiles `asked` names, when a made photograph has every one of them
    /// kept, each counted as asked for at `tick`: what a request is answered
    /// without a decode. For [`Asked::Photograph`], no tile, when a tile of
    /// the photograph is kept, each kept tile counted as asked for, and
    /// `slot`, this data's slot, for the photograph to read through. `None`
    /// when the photograph is not made, its decode failed, or a tile it asks
    /// for is not kept.
    fn kept(&mut self, asked: Asked<'_>, tick: u64, slot: &Arc<Slot>) -> Option<Got> {
        let Some(Ok(layout)) = &self.made else {
            return None;
        };
        let layout = Arc::clone(layout);
        let channels = self.channels;
        let tiles = match asked {
            Asked::Photograph => {
                if self.tiles.is_empty() {
                    return None;
                }
                for kept in self.tiles.values_mut() {
                    kept.tick = tick;
                }
                return Some(Got {
                    layout,
                    tiles: Vec::new(),
                    channels,
                    slot: Some(Arc::clone(slot)),
                });
            }
            Asked::Every => {
                if self.tiles.len() != tile_count(&layout) {
                    return None;
                }
                for kept in self.tiles.values_mut() {
                    kept.tick = tick;
                }
                self.tiles.values().map(|k| Arc::clone(&k.tile)).collect()
            }
            Asked::Keys(keys) => {
                let keys: Vec<TileKey> = keys
                    .iter()
                    .copied()
                    .filter(|&key| has_key(&layout, key))
                    .collect();
                // Every one is looked for before any is marked, so a request
                // that is not answered marks none.
                if !keys.iter().all(|key| self.tiles.contains_key(key)) {
                    return None;
                }
                keys.iter()
                    .map(|key| {
                        let kept = self.tiles.get_mut(key)?;
                        (kept.used, kept.tick) = (true, tick);
                        Some(Arc::clone(&kept.tile))
                    })
                    .collect::<Option<Vec<_>>>()?
            }
        };
        Some(Got {
            layout,
            tiles,
            channels,
            slot: None,
        })
    }

    /// Drop every tile kept, answering the bytes they held.
    fn drop_tiles(&mut self) -> u64 {
        self.tiles.clear();
        std::mem::take(&mut self.bytes)
    }
}

/// Whether `layout` has the tile `key`.
fn has_key(layout: &PyramidLayout, key: TileKey) -> bool {
    let level = usize::from(key.level);
    let (cols, rows) = layout.tiles(level);
    level < layout.levels() && key.col < cols && key.row < rows
}

/// The bytes of a tile's pixels, what it is charged.
fn tile_bytes(tile: &PhotographTile) -> u64 {
    tile.pixels.data().len() as u64
}

/// One path's entry. Shared between the map and every thread using it, so a
/// thread can decode into it with the map lock released.
///
/// Locks are taken map first: a caller that holds a slot's lock never takes
/// the map's.
struct Slot {
    data: Mutex<SlotData>,
    /// Woken when `making` is cleared.
    ready: Condvar,
    stamp: FileStamp,
}

impl Slot {
    fn new(stamp: FileStamp) -> Self {
        Self {
            data: Mutex::new(SlotData::default()),
            ready: Condvar::new(),
            stamp,
        }
    }

    fn lock(&self) -> MutexGuard<'_, SlotData> {
        // The lock is held only to read or change the slot's state and its
        // tiles, never across a decode, so nothing panics while it is held;
        // poisoning is ignored.
        self.data.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The decoding caller's hold on a slot it marked as being made, with its
/// share of the build count. The share goes back before the slot changes
/// state, so a waiter that decodes next never finds this decode still
/// counted. A hold dropped without [`PhotographCache::keep`] (the caller was
/// cancelled, or it unwinds from a panic) gives the share back, clears
/// `making` and wakes the slot's waiters, so no slot is left marked as being
/// made by a caller that has stopped, and the first waiter to take the lock
/// decodes in its place. What the slot held before stays.
struct Making<'s, 'r> {
    slot: &'s Arc<Slot>,
    share: Share<'r>,
    finished: bool,
}

impl Making<'_, '_> {
    /// Give the share back and mark the hold finished, so dropping it leaves
    /// the slot to its finisher.
    fn finish(&mut self) {
        self.finished = true;
        self.share.release();
    }
}

impl Drop for Making<'_, '_> {
    fn drop(&mut self) {
        if !self.finished {
            self.share.release();
            self.slot.lock().making = false;
            self.slot.ready.notify_all();
        }
    }
}

struct Inner {
    map: HashMap<PathBuf, Arc<Slot>>,
    /// The sum of every mapped slot's `bytes`.
    bytes: u64,
    /// The layouts [`PhotographCache::request_layout`] read, or why it could
    /// not, kept whatever the budget and not counted against it, since each is
    /// a few dozen bytes. `forget` and `clear` drop them.
    layouts: HashMap<PathBuf, Result<Arc<PyramidLayout>, String>>,
}

impl Inner {
    /// Take `path`'s slot out of the map, with its tiles.
    fn remove(&mut self, path: &Path) {
        if let Some(old) = self.map.remove(path) {
            self.bytes -= old.lock().drop_tiles();
        }
    }
}

/// Decoded photographs as photograph tiles, keyed by path, shared across
/// threads.
///
/// A decode is `ImageU8::read_rgb` of the file, cut into tiles of
/// [`PhotographCacheOptions::tile_size`] with their borders, level by level,
/// of the pyramid [`ImageU8Pyramid::from_image`] builds at
/// [`ImageU8Pyramid::full_levels`] of the photograph's own size, so each
/// photograph's depth is set by its image; the pyramid is never held whole,
/// and only the tiles are kept. The tiles a request asked for are kept as
/// used and the rest of the decode as unused. [`Self::get`] and
/// [`Self::get_many`] answer a [`TiledPhotograph`] that holds no tile and
/// reads each through the cache, which decodes the photograph again for a
/// tile it no longer keeps; they decode only a photograph none of whose tiles
/// is kept. Under a budget of 0 the cache keeps nothing, and the photograph
/// they answer holds every tile of the decode that made it. A photograph
/// with a side over [`MAX_JPEG_SIDE`] is refused from its header, before any
/// decode, as [`GetError::TooLarge`]. A file that cannot be read or is
/// refused is remembered as its error and costs nothing against the budget. An entry records the file's length and modification
/// time when it is made; `get` and `get_many` compare them with the file on
/// disk and decode again when they differ, so a remembered failure is also
/// retried once the file appears or changes.
///
/// When a decode takes the bytes kept over the budget, tiles are dropped
/// until the total is back under it: the unused before the used, the least
/// recently asked for first, and never one the decode's own request asked
/// for (every tile of a `get`'s decode). Only the cache's `Arc` is dropped: a
/// caller still holding a tile keeps its pixels alive, and a photograph `get`
/// answered reads a tile it needs that was dropped by decoding the file
/// again.
///
/// Each path has one slot. The caller that finds the tiles it asks for not
/// kept marks the slot as being made and decodes; the others wait on it,
/// looking at their own `Progress` every [`SLOT_WAIT_POLL`], and stop with
/// [`GetError::Cancelled`] when it is cancelled, leaving the decode to its
/// maker. A maker that is cancelled between the decode and cutting the tiles,
/// or that unwinds, leaves the slot as it found it, and a waiter decodes in
/// its place.
///
/// Every whole decode is first reserved on the cache's build count, an atomic
/// that belongs to this cache alone, against
/// [`PhotographCacheOptions::build_budget_bytes`]. Nothing waits on the
/// count: [`Self::reserve_builds`] grants at least one decode, so a caller
/// always makes progress, and [`Self::try_reserve_build`] is refused when its
/// decode does not fit while another decode is reserved.
///
/// [`ImageU8Pyramid::from_image`]: crate::camera::image::ImageU8Pyramid::from_image
/// [`ImageU8Pyramid::full_levels`]: crate::camera::image::ImageU8Pyramid::full_levels
pub struct PhotographCache {
    /// What the cache keeps and counts, shared with every photograph
    /// [`Self::get`] and [`Self::get_many`] answer that reads its tiles
    /// through the cache.
    shared: Arc<Shared>,
}

/// A [`PhotographCache`]'s state: its tiles, its counters and its build
/// count. Held in an `Arc` by the cache and by each photograph it answers
/// that reads through it, so such a photograph stays readable after the
/// cache that answered it is dropped, keeping the tiles kept until then.
struct Shared {
    options: PhotographCacheOptions,
    inner: Mutex<Inner>,
    tick: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    decodes: AtomicU64,
    /// Estimated bytes of the whole decodes reserved now.
    build_reserved: AtomicU64,
    /// Called as a decode starts, once its header is read and the decode is
    /// reserved and before the pixels are read, as a caller starts to wait on
    /// a slot being made, and as a panel's decode has its reservation and is
    /// about to claim the slot: what lets a test hold a decode in progress,
    /// cancel a caller in the middle of a decode or a wait, make a decode
    /// panic, change a slot under a panel's decode, or read the build count at
    /// those moments.
    #[cfg(test)]
    on_slot_event: Option<Box<SlotHook>>,
}

/// The test build's hook on a cache's slot events, told the cache too so it
/// can read the build count.
#[cfg(test)]
pub(crate) type SlotHook = dyn Fn(&PhotographCache, SlotEvent) + Send + Sync;

/// What [`PhotographCache::on_slot_event`] is told.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SlotEvent {
    DecodeStarted,
    WaitStarted,
    /// A panel's decode ([`PhotographCache::try_decode_tiles`]) has its
    /// reservation and is about to claim the slot.
    PanelReserved,
}

impl std::fmt::Debug for PhotographCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhotographCache")
            .field("stats", &self.stats())
            .finish()
    }
}

/// Whether a request decoded the photograph itself.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Found {
    Decoded,
    Reused,
}

/// One path's answer inside `get_many`.
type Answer = (Result<Arc<TiledPhotograph>, GetError>, Found);

impl PhotographCache {
    /// A cache keeping up to `options.budget_bytes` of photograph tiles, cut
    /// from pyramids each of [`ImageU8Pyramid::full_levels`] depth for its
    /// photograph's size, with the whole decodes in progress on it bounded by
    /// `options.build_budget_bytes`. A budget of 0 keeps nothing: every call
    /// decodes and hands back tiles no one else will see. A build budget of 0
    /// lets each caller decode one photograph at a time.
    ///
    /// [`ImageU8Pyramid::full_levels`]: crate::camera::image::ImageU8Pyramid::full_levels
    pub fn new(options: PhotographCacheOptions) -> Self {
        Self {
            shared: Arc::new(Shared {
                options,
                inner: Mutex::new(Inner {
                    map: HashMap::new(),
                    bytes: 0,
                    layouts: HashMap::new(),
                }),
                tick: AtomicU64::new(0),
                hits: AtomicU64::new(0),
                misses: AtomicU64::new(0),
                decodes: AtomicU64::new(0),
                build_reserved: AtomicU64::new(0),
                #[cfg(test)]
                on_slot_event: None,
            }),
        }
    }

    /// Another handle on the same cache, which a photograph that reads
    /// through the cache holds.
    fn handle(&self) -> PhotographCache {
        PhotographCache {
            shared: Arc::clone(&self.shared),
        }
    }

    /// Set the test build's hook on the cache's slot events, before any
    /// photograph holds a handle on the cache.
    #[cfg(test)]
    pub(crate) fn set_slot_hook(&mut self, hook: Box<SlotHook>) {
        Arc::get_mut(&mut self.shared)
            .expect("a cache no photograph holds yet")
            .on_slot_event = Some(hook);
    }

    /// The budget the cache was made with. At 0 it keeps nothing, so a decode
    /// that one caller starts is never there for another to [`Self::peek`].
    pub fn budget_bytes(&self) -> u64 {
        self.shared.options.budget_bytes
    }

    /// Reserve up to `wanted` whole decodes of `bytes_per_build` estimated
    /// bytes each against this cache's build budget, at the moment the caller
    /// decides, with one compare-exchange loop on its count; never waits.
    /// Grants as many as fit under the budget beside what other callers have
    /// reserved, and one when none fit (a decode larger than the whole budget
    /// included), so a caller always makes progress; that one decode is the
    /// only way the count goes past the budget. Grants none only when
    /// `wanted` is 0.
    pub fn reserve_builds(&self, bytes_per_build: u64, wanted: usize) -> BuildReservation<'_> {
        let budget = self.shared.options.build_budget_bytes;
        // How many to grant when `reserved` bytes are reserved already.
        let grant = |reserved: u64| {
            let free = budget.saturating_sub(reserved);
            let fit = match free.checked_div(bytes_per_build) {
                Some(fit) => usize::try_from(fit).unwrap_or(usize::MAX).min(wanted),
                // Decodes of no bytes all fit.
                None => wanted,
            };
            fit.max(wanted.min(1))
        };
        let before = update(&self.shared.build_reserved, |reserved| {
            let bytes = bytes_per_build.saturating_mul(grant(reserved) as u64);
            Some(reserved.saturating_add(bytes))
        });
        // `update` always sets the count here, since the closure never
        // answers `None`; the grant is the one made from the count it found.
        let granted = before.map_or(0, grant);
        BuildReservation {
            count: &self.shared.build_reserved,
            bytes_per_build,
            granted,
            released: AtomicU64::new(0),
        }
    }

    /// Reserve one whole decode of `bytes` estimated bytes if it fits under
    /// this cache's build budget beside what other callers have reserved, or
    /// if nothing at all is reserved: `None` otherwise. So a decode larger
    /// than the whole budget, every decode under a build budget of 0, runs
    /// alone when the count is idle, as [`Self::reserve_builds`] grants one
    /// decode when none fits, and the count passes the budget only by that
    /// one decode. Never waits; a caller that is refused asks again later.
    pub fn try_reserve_build(&self, bytes: u64) -> Option<BuildReservation<'_>> {
        let budget = self.shared.options.build_budget_bytes;
        update(&self.shared.build_reserved, |reserved| {
            let after = reserved.saturating_add(bytes);
            (reserved == 0 || after <= budget).then_some(after)
        })?;
        Some(BuildReservation {
            count: &self.shared.build_reserved,
            bytes_per_build: bytes,
            granted: 1,
            released: AtomicU64::new(0),
        })
    }

    /// The photograph at `path`, which holds its layout and the cache's handle
    /// on its tiles and no pixel, decoding it into tiles when none of them is
    /// kept. Waits while another thread is decoding
    /// the same path, and then answers from its decode rather than decoding
    /// a second time; if that thread is cancelled or unwinds, the first
    /// waiter decodes in its place. An error when the file does not exist,
    /// cannot be read, or has a side over [`MAX_JPEG_SIDE`]; that answer is
    /// remembered until `forget` or `clear`, or until the file's length or
    /// modification time changes.
    ///
    /// A decode keeps every tile as unused, so tiles a panel or a kernel's
    /// tile read asks for are evicted after them, and never evicts one of its
    /// own tiles. The photograph reads each tile through the cache, as
    /// [`Self::get_tile`] does: one the cache still keeps it takes, marking
    /// it used, and one the cache has evicted it reads by decoding the
    /// photograph again, reserving with [`Self::reserve_builds`], waiting
    /// while another caller decodes it, and polling the `Progress` of the
    /// render that reads it. Such a read fails with
    /// [`PhotographReadError::NotFound`] when the file is gone,
    /// [`PhotographReadError::Changed`] when its length, modification time or
    /// size are not the ones the photograph was decoded from, and the error
    /// of a decode that fails otherwise. The photograph holds the cache's
    /// state alive, so it stays readable after the cache is dropped.
    ///
    /// Under a budget of 0 the cache keeps nothing, so the photograph holds
    /// every tile of the decode that made it while the caller holds it, and
    /// its reads decode nothing.
    ///
    /// Polls `progress` for cancellation while it waits, every
    /// [`SLOT_WAIT_POLL`], and after its own decode, before it cuts the
    /// tiles, and answers [`GetError::Cancelled`] when it is cancelled,
    /// remembering nothing. It reports nothing through `progress`; a caller
    /// with nothing to cancel passes `&Progress::none()`.
    ///
    /// A decode it makes is reserved with [`Self::reserve_builds`] when it has
    /// marked the slot as being made, whether its tiles were not kept or a
    /// stopped maker has just left it, and given back when the decode ends,
    /// is cancelled or unwinds. It holds no share while it waits.
    ///
    /// A caller already inside a rayon parallel loop calls this rather than
    /// [`Self::get_many`].
    pub fn get(
        &self,
        path: &Path,
        progress: &Progress<'_>,
    ) -> Result<Arc<TiledPhotograph>, GetError> {
        self.get_found(path, progress, Share::None).0
    }

    /// The photograph at `path` if every tile of it is kept, without reading
    /// anything and without waiting on a decode in flight. A caller that only
    /// asks whether it is held calls [`Self::keeps_every_tile`], which builds
    /// no photograph.
    ///
    /// It does not compare the entry with the file on disk, and counts as a
    /// use of every tile for the eviction order but not as a hit.
    pub fn peek(&self, path: &Path) -> Option<Arc<TiledPhotograph>> {
        match self.peek_state(path) {
            PeekedPhotograph::Decoded(photograph) => Some(photograph),
            PeekedPhotograph::Unreadable | PeekedPhotograph::NotDecoded => None,
        }
    }

    /// [`Self::peek`], telling a remembered failure apart from a path with no
    /// answer yet: what a caller that starts a decode on a miss needs, so that
    /// it does not start one again for a file that cannot be read.
    ///
    /// Like `peek` it reads nothing, waits on nothing and does not stat the
    /// file, so a remembered failure is reported even when the file has since
    /// appeared; the next `get` notices that.
    pub fn peek_state(&self, path: &Path) -> PeekedPhotograph {
        let Some(slot) = self.mapped(path) else {
            return PeekedPhotograph::NotDecoded;
        };
        let tick = self.next_tick();
        let mut data = slot.lock();
        if let Some(Err(_)) = data.made {
            return PeekedPhotograph::Unreadable;
        }
        let Some(got) = data.kept(Asked::Every, tick, &slot) else {
            return PeekedPhotograph::NotDecoded;
        };
        drop(data);
        match got.photograph(self, path) {
            Ok(photograph) => PeekedPhotograph::Decoded(photograph),
            Err(_) => PeekedPhotograph::NotDecoded,
        }
    }

    /// Whether the cache keeps every tile of the photograph at `path`, so
    /// [`Self::peek`] would answer it: what a caller that only asks whether
    /// a photograph is held, such as the image menu on every frame, calls.
    ///
    /// It reads nothing, waits on nothing and does not stat the file, and
    /// unlike `peek` it allocates nothing and is no use for the eviction
    /// order: it leaves every tile's place in that order as it was. It
    /// compares the number of tiles kept with the layout's, a few
    /// arithmetic steps a level.
    pub fn keeps_every_tile(&self, path: &Path) -> bool {
        let Some(slot) = self.mapped(path) else {
            return false;
        };
        let data = slot.lock();
        matches!(&data.made, Some(Ok(layout)) if data.tiles.len() == tile_count(layout))
    }

    /// `get` for every path, the misses decoded in parallel on the rayon
    /// pool. Returns one entry per path, in order, and the tally the caller's
    /// phase note reports.
    ///
    /// Each photograph it answers is one `get` answers, so the answers hold no
    /// tile and the call holds no pyramid: what it decodes stays in the cache,
    /// within its budget, and a render of an answer whose tiles were evicted
    /// decodes it again.
    ///
    /// It first looks for each path in the cache and reads the header of each
    /// one none of whose tiles it finds kept, to know its misses and their
    /// sizes. It then decodes the misses in groups, one parallel loop a
    /// group and the groups one after another, each group as many as
    /// [`Self::reserve_builds`] grants when it is decided, at most the rayon
    /// pool's size and at least one. A member that finds its path being
    /// decoded by another caller keeps its share while it waits, since it may
    /// take the decode over, and gives it back when it ends.
    ///
    /// Counts `images` through `progress` as each path is answered, and polls
    /// it for cancellation before each path and as [`Self::get`] polls it. It
    /// opens no phase of its own:
    /// the caller names the phase (the viewer's is `decode images`) and writes
    /// its note from the tally, and may add [`Self::stats`] to it.
    ///
    /// # Errors
    ///
    /// [`Cancelled`] when `progress` was cancelled. What was decoded before
    /// that stays in the cache; a decode it stopped leaves its slot as it
    /// was.
    // Spelled out rather than behind an alias, so the signature reads the same
    // here, in the docs and in the spec.
    #[allow(clippy::type_complexity)]
    pub fn get_many(
        &self,
        paths: &[&Path],
        progress: &Progress<'_>,
    ) -> Result<(Vec<Option<Arc<TiledPhotograph>>>, GetManyTally), Cancelled> {
        let total = paths.len();
        let answered = AtomicUsize::new(0);
        let answer = |found| {
            let n = answered.fetch_add(1, Ordering::Relaxed) + 1;
            progress.count(n as u64, Some(total as u64), "images");
            Some(found)
        };
        // A path made from the file as it is now, with every tile kept, is
        // answered here; any other is a miss, with the estimated bytes of its
        // decode, 0 when its header cannot be read (its `get` then stops at
        // the header too, decoding nothing).
        let looked: Vec<Result<Option<Answer>, u64>> = paths
            .par_iter()
            .map(|path| {
                if progress.is_cancelled() {
                    return Ok(None);
                }
                match self.made_now(path) {
                    Some(value) => Ok(answer((value, Found::Reused))),
                    None => Err(image_dimensions(path)
                        .map_or(0, |(width, height)| whole_decode_bytes(width, height))),
                }
            })
            .collect();
        let mut found = Vec::with_capacity(total);
        let mut misses = Vec::new();
        for (i, looked) in looked.into_iter().enumerate() {
            match looked {
                Ok(value) => found.push(value),
                Err(bytes) => {
                    found.push(None);
                    misses.push((i, bytes));
                }
            }
        }
        // The misses in groups, each reserved as it is decided. The group's
        // bytes per decode are its largest photograph's, so the count never
        // holds less than the group's decodes.
        let pool = rayon::current_num_threads().max(1);
        let mut rest = &misses[..];
        while !rest.is_empty() && !progress.is_cancelled() {
            let candidates = &rest[..rest.len().min(pool)];
            let bytes_per_build = candidates.iter().map(|&(_, bytes)| bytes).max();
            let reservation = self.reserve_builds(bytes_per_build.unwrap_or(0), candidates.len());
            let (group, after) = rest.split_at(reservation.granted());
            rest = after;
            let made: Vec<(usize, Option<Answer>)> = group
                .par_iter()
                .map(|&(i, _)| {
                    let share = Share::Group(GroupShare(&reservation));
                    if progress.is_cancelled() {
                        return (i, None);
                    }
                    (i, answer(self.get_found(paths[i], progress, share)))
                })
                .collect();
            for (i, value) in made {
                found[i] = value;
            }
        }
        progress.check_cancel()?;
        let mut tally = GetManyTally::default();
        let photographs = found
            .into_iter()
            .map(|entry| {
                // Every entry is `Some`, and none is `GetError::Cancelled`,
                // which only a cancelled `progress` gives, when the call was
                // not cancelled.
                let (photograph, how) = entry.ok_or(Cancelled)?;
                match (&photograph, how) {
                    (Err(_), _) => tally.unreadable += 1,
                    (Ok(_), Found::Decoded) => tally.read += 1,
                    (Ok(_), Found::Reused) => tally.reused += 1,
                }
                Ok(photograph.ok())
            })
            .collect::<Result<Vec<_>, Cancelled>>()?;
        Ok((photographs, tally))
    }

    /// Put `image` in the cache as the decode of `path`, cut into tiles as a
    /// decode cuts them and kept as unused, replacing any entry for it, and
    /// drop tiles down to the budget as a decode would, never one of these. A
    /// cache with a budget of 0 keeps nothing, so there this does nothing.
    ///
    /// For a caller whose pixels for `path` did not come from this cache's
    /// decode, such as a test fixture standing in for a photograph that is not
    /// on disk. The entry records the file's length and modification time as
    /// they are now (or that there is no file), so `get` answers with the
    /// photograph of these tiles until the file changes, or until one of them
    /// is evicted, when it decodes the file.
    pub fn insert(&self, path: &Path, image: ImageU8) {
        if self.shared.options.budget_bytes == 0 {
            return;
        }
        let cut = cut_image(image, self.shared.options.tile_size);
        let slot = Arc::new(Slot::new(file_stamp(path)));
        let mut inner = self.lock();
        inner.remove(path);
        inner.map.insert(path.to_path_buf(), Arc::clone(&slot));
        // The answer is the tiles just kept, which `insert` hands no one.
        let _ = self.store(&mut inner, path, &slot, Ok(cut), Asked::Every);
    }

    /// Drop one path's entry, its tiles and its layout. The next `get` of it
    /// reads the file again.
    pub fn forget(&self, path: &Path) {
        let mut inner = self.lock();
        inner.remove(path);
        inner.layouts.remove(path);
    }

    /// Drop every entry, every tile and every layout.
    pub fn clear(&self) {
        let mut inner = self.lock();
        for slot in inner.map.values() {
            slot.lock().drop_tiles();
        }
        inner.map.clear();
        inner.layouts.clear();
        inner.bytes = 0;
    }

    /// Tiles and bytes kept, the budget, the hit, miss and decode counters,
    /// the build budget and the bytes reserved on the build count.
    pub fn stats(&self) -> PhotographCacheStats {
        let inner = self.lock();
        let tiles = inner.map.values().map(|slot| slot.lock().tiles.len()).sum();
        PhotographCacheStats {
            tiles,
            bytes: inner.bytes,
            budget_bytes: self.shared.options.budget_bytes,
            hits: self.shared.hits.load(Ordering::Relaxed),
            misses: self.shared.misses.load(Ordering::Relaxed),
            decodes: self.shared.decodes.load(Ordering::Relaxed),
            build_budget_bytes: self.shared.options.build_budget_bytes,
            build_reserved_bytes: self.shared.build_reserved.load(Ordering::Acquire),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        // Nothing panics while the lock is held except on a broken invariant,
        // and the map is still usable then, so poisoning is ignored.
        self.shared.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[cfg(test)]
    fn slot_event(&self, event: SlotEvent) {
        if let Some(on_slot_event) = &self.shared.on_slot_event {
            on_slot_event(self, event);
        }
    }

    /// The next tick of the eviction order.
    fn next_tick(&self) -> u64 {
        self.shared.tick.fetch_add(1, Ordering::Relaxed)
    }

    /// `path`'s slot when it is in the map, without a stat of the file.
    fn mapped(&self, path: &Path) -> Option<Arc<Slot>> {
        self.lock().map.get(path).map(Arc::clone)
    }

    /// Decode `path` and cut its tiles, polling `progress` between the two:
    /// [`GetError::Cancelled`] when it was cancelled during the decode.
    /// Reads the header first and, when `share` holds nothing, reserves the
    /// decode with [`Self::reserve_builds`] before reading the pixels.
    ///
    /// The image is dropped once its level-0 tiles are cut, and each next
    /// level is made from the tiles of the one before, so the cutting holds
    /// at its peak the image and its level-0 tiles, about 2.0 times the
    /// image's bytes, and never the whole pyramid with its tiles; the
    /// decoder's own peak, inside `read_rgb`, is higher.
    ///
    /// Must not use rayon. It runs while its slot is marked as being made, and
    /// a rayon worker waiting in a nested join may steal another `get_many`
    /// item for the same path, which would find that slot being made by its
    /// own thread and wait on itself forever. Neither the file decode nor the
    /// cutting uses rayon's global pool. The decode may start threads of its
    /// own (one per colour component in `jpeg-decoder`, a private pool in
    /// `exr`), which take no work from the global pool; see the
    /// photograph-cache spec.
    fn decode<'r>(
        &'r self,
        path: &Path,
        progress: &Progress<'_>,
        share: &mut Share<'r>,
    ) -> Result<CutImage, GetError> {
        let (width, height) = header(path)?;
        if let Share::None = share {
            *share = Share::Own(self.reserve_builds(whole_decode_bytes(width, height), 1));
        }
        #[cfg(test)]
        self.slot_event(SlotEvent::DecodeStarted);
        let image = ImageU8::read_rgb(path).map_err(|e| unreadable(path, e))?;
        self.shared.decodes.fetch_add(1, Ordering::Relaxed);
        if progress.is_cancelled() {
            return Err(GetError::Cancelled);
        }
        Ok(cut_image(image, self.shared.options.tile_size))
    }

    /// The photograph at `path` when its slot is made from the file as it is
    /// now and keeps a tile of it, or the failure it remembers, counted as a
    /// hit; `None` otherwise. Reads nothing but the file's stamp and
    /// never waits on a decode.
    fn made_now(&self, path: &Path) -> Option<Result<Arc<TiledPhotograph>, GetError>> {
        if self.shared.options.budget_bytes == 0 {
            return None;
        }
        let stamp = file_stamp(path);
        let slot = match self.lock().map.get(path) {
            Some(slot) if slot.stamp == stamp => Arc::clone(slot),
            _ => return None,
        };
        let tick = self.next_tick();
        let mut data = slot.lock();
        if data.making {
            return None;
        }
        let value = match &data.made {
            Some(Err(e)) => Err(e.clone()),
            _ => Ok(data.kept(Asked::Photograph, tick, &slot)?),
        };
        drop(data);
        self.shared.hits.fetch_add(1, Ordering::Relaxed);
        Some(value.and_then(|got| got.photograph(self, path)))
    }

    /// `path`'s slot made from the file as it is now: the mapped one when its
    /// stamp matches the file's, otherwise a new empty slot in its place.
    fn slot_for(&self, path: &Path) -> Arc<Slot> {
        // Stat with the lock released; it is a system call.
        let stamp = file_stamp(path);
        let mut inner = self.lock();
        match inner.map.get(path) {
            Some(slot) if slot.stamp == stamp => Arc::clone(slot),
            _ => {
                inner.remove(path);
                let slot = Arc::new(Slot::new(stamp));
                inner.map.insert(path.to_path_buf(), Arc::clone(&slot));
                slot
            }
        }
    }

    /// `get`, telling whether it decoded. `share` is the caller's share of
    /// the build count when it already holds one (a `get_many` member's), or
    /// [`Share::None`] for a `get`, which reserves when it decodes. Whatever
    /// the outcome, the share has been given back when this returns.
    fn get_found<'r>(
        &'r self,
        path: &Path,
        progress: &Progress<'_>,
        share: Share<'r>,
    ) -> (Result<Arc<TiledPhotograph>, GetError>, Found) {
        let (value, found) = self.fetch(path, progress, share, Asked::Photograph);
        (value.and_then(|got| got.photograph(self, path)), found)
    }

    /// The tiles `asked` names of `path`, decoding the photograph when one of
    /// them is not kept, as [`Self::get`] describes, and telling whether it
    /// decoded. `share` is as for [`Self::get_found`].
    fn fetch<'r>(
        &'r self,
        path: &Path,
        progress: &Progress<'_>,
        mut share: Share<'r>,
        asked: Asked<'_>,
    ) -> (Result<Got, GetError>, Found) {
        if self.shared.options.budget_bytes == 0 {
            self.shared.misses.fetch_add(1, Ordering::Relaxed);
            let value = self.decode(path, progress, &mut share);
            return (value.map(|cut| answer_of(cut, asked)), Found::Decoded);
        }
        let slot = self.slot_for(path);

        // Wait while another caller makes the slot, and take it over when
        // what is asked for is not kept, as before anyone has decoded it,
        // after a tile of it was evicted and after its maker stopped.
        #[cfg(test)]
        let mut told_waiting = false;
        let mut making = {
            let mut data = slot.lock();
            loop {
                if !data.making {
                    if let Some(Err(e)) = &data.made {
                        self.shared.hits.fetch_add(1, Ordering::Relaxed);
                        return (Err(e.clone()), Found::Reused);
                    }
                    if let Some(got) = data.kept(asked, self.next_tick(), &slot) {
                        self.shared.hits.fetch_add(1, Ordering::Relaxed);
                        return (Ok(got), Found::Reused);
                    }
                    data.making = true;
                    break Making {
                        slot: &slot,
                        share,
                        finished: false,
                    };
                }
                if progress.is_cancelled() {
                    return (Err(GetError::Cancelled), Found::Reused);
                }
                #[cfg(test)]
                if !std::mem::replace(&mut told_waiting, true) {
                    self.slot_event(SlotEvent::WaitStarted);
                }
                data = slot
                    .ready
                    .wait_timeout(data, SLOT_WAIT_POLL)
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
        };
        self.shared.misses.fetch_add(1, Ordering::Relaxed);
        let value = self.decode(path, progress, &mut making.share);
        if let Err(GetError::Cancelled) = value {
            // Dropping the hold leaves the slot as it was and wakes its
            // waiters.
            drop(making);
            return (value.map(|cut| answer_of(cut, asked)), Found::Decoded);
        }
        (self.keep(path, making, value, asked), Found::Decoded)
    }

    /// Finish the decode `making` holds `path`'s slot for: give its share
    /// back, put `value` in the slot, keeping its tiles as [`Self::store`]
    /// does, and wake the slot's waiters. Answers the tiles `asked` names.
    fn keep(
        &self,
        path: &Path,
        mut making: Making<'_, '_>,
        value: Result<CutImage, GetError>,
        asked: Asked<'_>,
    ) -> Result<Got, GetError> {
        making.finish();
        let slot = making.slot;
        let answer = {
            let mut inner = self.lock();
            self.store(&mut inner, path, slot, value, asked)
        };
        slot.ready.notify_all();
        answer
    }

    /// Put a decode's tiles, or its failure, in `slot`, `path`'s, clearing
    /// `making`, and answer the tiles `asked` names. The tiles are kept only
    /// while the slot is in the map, where a tile already kept stays and the
    /// decode's copy of it is dropped, so a caller holding it and the cache
    /// share one; the tiles `asked` names are kept as used, the others as
    /// unused, and the cache then drops tiles down to its budget, never one
    /// `asked` names. A slot that left the map while its decode ran
    /// (forgotten, cleared or replaced by a newer file) keeps no tile: its
    /// caller still gets its tiles, but the cache neither keeps nor counts
    /// them. A failure drops every tile the slot kept.
    fn store(
        &self,
        inner: &mut Inner,
        path: &Path,
        slot: &Arc<Slot>,
        value: Result<CutImage, GetError>,
        asked: Asked<'_>,
    ) -> Result<Got, GetError> {
        let mapped = inner
            .map
            .get(path)
            .is_some_and(|mapped| Arc::ptr_eq(mapped, slot));
        let mut data = slot.lock();
        data.making = false;
        let cut = match value {
            Ok(cut) => cut,
            Err(e) => {
                let freed = data.drop_tiles();
                if mapped {
                    inner.bytes -= freed;
                }
                data.made = Some(Err(e.clone()));
                return Err(e);
            }
        };
        // The layout already made, when it is the same, so a panel asking
        // every frame is answered the same `Arc`.
        let layout = match &data.made {
            Some(Ok(made)) if **made == cut.layout => Arc::clone(made),
            _ => Arc::new(cut.layout),
        };
        data.made = Some(Ok(Arc::clone(&layout)));
        if let Some(tile) = cut.tiles.values().next() {
            data.channels = tile.pixels.channels();
        }
        if !mapped {
            return Ok(answer_of(
                CutImage {
                    layout: (*layout).clone(),
                    tiles: cut.tiles,
                },
                asked,
            ));
        }
        let tick = self.next_tick();
        let used = |key: &TileKey| match asked {
            Asked::Photograph | Asked::Every => false,
            Asked::Keys(keys) => keys.contains(key),
        };
        let mut added = 0;
        for (key, tile) in cut.tiles {
            let used = used(&key);
            let kept = data.tiles.entry(key).or_insert_with(|| {
                added += tile_bytes(&tile);
                Kept {
                    tile,
                    used: false,
                    tick,
                }
            });
            kept.tick = tick;
            kept.used |= used;
        }
        data.bytes += added;
        inner.bytes += added;
        let got = data.kept(asked, tick, slot);
        drop(data);
        self.evict(inner, slot, asked);
        // Every tile of the layout was put in the slot just above.
        got.ok_or_else(|| GetError::Unreadable(format!("{}: tiles not kept", path.display())))
    }

    /// Drop tiles until the bytes kept are within the budget: the unused
    /// before the used, the least recently asked for first and, among tiles
    /// asked for together, the finer levels first. Never one of `slot`'s
    /// tiles that `asked` names (every one of them for [`Asked::Photograph`]
    /// and [`Asked::Every`]): those the request that just kept them is
    /// answered, or reads first.
    fn evict(&self, inner: &mut Inner, slot: &Arc<Slot>, asked: Asked<'_>) {
        if inner.bytes <= self.shared.options.budget_bytes {
            return;
        }
        let slots: Vec<Arc<Slot>> = inner.map.values().map(Arc::clone).collect();
        let mut candidates: Vec<(bool, u64, u8, u32, u32, usize)> = Vec::new();
        for (s, other) in slots.iter().enumerate() {
            let own = Arc::ptr_eq(other, slot);
            for (key, kept) in other.lock().tiles.iter() {
                let spared = own
                    && match asked {
                        Asked::Photograph | Asked::Every => true,
                        Asked::Keys(keys) => keys.contains(key),
                    };
                if !spared {
                    candidates.push((kept.used, kept.tick, key.level, key.row, key.col, s));
                }
            }
        }
        candidates.sort_unstable();
        for (_, _, level, row, col, s) in candidates {
            if inner.bytes <= self.shared.options.budget_bytes {
                break;
            }
            let mut data = slots[s].lock();
            let key = TileKey { level, col, row };
            if let Some(kept) = data.tiles.remove(&key) {
                let bytes = tile_bytes(&kept.tile);
                data.bytes -= bytes;
                inner.bytes -= bytes;
            }
        }
    }
}

/// The tiles `asked` names of a decode the cache does not keep: every tile,
/// which a photograph `get` answers then holds, or those of `keys` the layout
/// has.
fn answer_of(cut: CutImage, asked: Asked<'_>) -> Got {
    let CutImage { layout, tiles } = cut;
    let channels = tiles.values().next().map_or(0, |t| t.pixels.channels());
    let tiles = match asked {
        Asked::Photograph | Asked::Every => tiles.into_values().collect(),
        Asked::Keys(keys) => keys
            .iter()
            .filter_map(|key| tiles.get(key).cloned())
            .collect(),
    };
    Got {
        layout: Arc::new(layout),
        tiles,
        channels,
        slot: None,
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod read_tests;
