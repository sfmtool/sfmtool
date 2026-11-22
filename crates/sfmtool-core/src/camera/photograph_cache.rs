// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Decoded photographs as image pyramids, keyed by file path, shared across
//! threads and bounded by a byte budget, with the whole decodes in progress
//! bounded by a build budget.
//!
//! Decoding a photograph and building the pyramid the patch samplers read is
//! the expensive part of every operation that reads a reconstruction's pixels.
//! A [`PhotographCache`] keeps those pyramids once decoded, so that whichever
//! thread asks next for the same file gets the same pixels without reading it
//! again, and drops the least recently used ones when the bytes it holds pass
//! its budget.
//!
//! Its callers hold it in an `Arc` and call it through `&self`: a worker thread
//! puts what it decodes where the next worker and the GUI thread find it.
//! [`PhotographCache::get`] waits while another thread decodes the same path,
//! and stops waiting when its caller is cancelled;
//! [`PhotographCache::peek`] never reads a file and never waits, which is what a
//! caller that draws every frame wants. Every whole decode is reserved on the
//! cache's build count first ([`PhotographCache::reserve_builds`]), so the
//! decodes in progress on one cache stay within its build budget. The design
//! is in the spec `specs/core/camera/photograph-cache.md`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, SystemTime};

use rayon::prelude::*;

use crate::camera::image::{image_dimensions, ImageU8, ImageU8Pyramid};
use crate::camera::mipmap::{MAX_JPEG_SIDE, PHOTOGRAPH_TILE_SIZE};
use crate::progress::{Cancelled, Progress};

/// How long a caller waiting for another caller's decode of the same path
/// waits on the slot before it looks at its own `Progress` for cancellation
/// again: a cancelled waiter stops within about this long.
pub const SLOT_WAIT_POLL: Duration = Duration::from_millis(50);

/// The environment variable that overrides [`default_budget_bytes`], in
/// megabytes (10^6 bytes), written as a decimal integer.
pub const BUDGET_ENV_VAR: &str = "SFM_EXPLORER_PHOTOGRAPH_CACHE_MB";

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

/// [`default_budget_bytes`] with its two inputs passed in, so it can be tested
/// without touching the environment.
fn budget_from(env_megabytes: Option<&str>, physical_bytes: Option<u64>) -> u64 {
    if let Some(mb) = env_megabytes.and_then(|v| v.trim().parse::<u64>().ok()) {
        return mb.saturating_mul(1_000_000);
    }
    match physical_bytes {
        Some(bytes) => (bytes / 4).clamp(GIB, 16 * GIB),
        None => 4 * GIB,
    }
}

/// The build budget a cache gets by default: a quarter of physical memory, or
/// 4 GiB where physical memory cannot be read. A pure function, like
/// [`default_budget_bytes`]; it keeps no state, and no environment variable
/// overrides it.
pub fn default_build_budget_bytes() -> u64 {
    build_budget_from(crate::machine::physical_memory_bytes())
}

/// [`default_build_budget_bytes`] with physical memory passed in, so it can be
/// tested on any machine.
fn build_budget_from(physical_bytes: Option<u64>) -> u64 {
    match physical_bytes {
        Some(bytes) => bytes / 4,
        None => 4 * GIB,
    }
}

/// The bytes a whole decode of a photograph is estimated to hold while it
/// runs, per pixel of the photograph: its RGB pixels (3) and the pyramid
/// levels built from them (about 1), with the decoder's own buffers on top.
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
    /// Bytes of decoded photographs held before the least recently used are
    /// dropped; 0 keeps nothing. By default [`default_budget_bytes`].
    pub budget_bytes: u64,
    /// Estimated bytes of the whole decodes in progress on this cache at
    /// once, what [`PhotographCache::reserve_builds`] and
    /// [`PhotographCache::try_reserve_build`] reserve against. By default
    /// [`default_build_budget_bytes`].
    pub build_budget_bytes: u64,
    /// The side of a photograph tile, by default [`PHOTOGRAPH_TILE_SIZE`];
    /// only tests change it. The cache holds whole pyramids and does not read
    /// it yet.
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

/// Why [`PhotographCache::get`] has no pyramid for a path.
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
    /// Decoded pyramids held. Remembered failures and decodes still in flight
    /// are not counted.
    pub entries: usize,
    /// Pixel bytes of the pyramids held ([`ImageU8Pyramid::byte_len`]).
    pub bytes: u64,
    /// The budget the cache was made with.
    pub budget_bytes: u64,
    /// `get` answers that decoded nothing: the entry was there, or another
    /// thread's decode of the same path was waited on. Each path of a
    /// `get_many` counts as one `get`. `peek` is not counted, and neither is
    /// a `get` that stopped waiting because it was cancelled.
    pub hits: u64,
    /// Decodes started, whether they succeeded, failed or were cancelled.
    pub misses: u64,
    /// Photographs decoded whole: every decode that read a photograph's
    /// pixels, including one whose caller was cancelled before its pyramid
    /// was built. A file whose header cannot be read, or that is refused as
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
    /// Already in the cache, or decoded by another thread while this call
    /// waited for it.
    pub reused: usize,
    /// Could not be read or decoded, now or when last tried.
    pub unreadable: usize,
}

/// What [`PhotographCache::peek_state`] found for a path: [`PhotographCache::peek`]
/// with a failure told apart from a photograph not yet decoded.
#[derive(Clone)]
pub enum PeekedPhotograph {
    /// The decoded pyramid.
    Decoded(Arc<ImageU8Pyramid>),
    /// A decode was tried and the file could not be read or decoded. The
    /// answer `get` gives until the entry is forgotten or the file changes.
    Unreadable,
    /// No answer yet: the path was never asked for, its entry was dropped, or
    /// its decode is still running.
    NotDecoded,
}

/// A file's length and modification time, as a slot records them when it is
/// made. `None` when the file cannot be stat'ed.
type FileStamp = Option<(u64, Option<SystemTime>)>;

fn file_stamp(path: &Path) -> FileStamp {
    let meta = std::fs::metadata(path).ok()?;
    Some((meta.len(), meta.modified().ok()))
}

/// Where a slot's decode stands.
enum SlotState {
    /// Nothing decoded and no one decoding: the next caller decodes. A slot
    /// starts so, and goes back to it when its decoding caller is cancelled
    /// or unwinds.
    Empty,
    /// One caller is decoding; the others wait on [`Slot::ready`].
    Making,
    /// The pyramid, or the failure remembered for the file.
    Made(Result<Arc<ImageU8Pyramid>, GetError>),
}

/// One path's entry. Shared between the map and every thread using it, so a
/// thread can decode into it with the map lock released.
struct Slot {
    state: Mutex<SlotState>,
    /// Woken when the state leaves [`SlotState::Making`].
    ready: Condvar,
    stamp: FileStamp,
    /// The cache's tick when the slot was last asked for, for LRU order.
    last_used: AtomicU64,
    /// Bytes this slot added to [`Inner::bytes`]. Read and written only under
    /// the map lock; 0 until the decoding thread charges it, and 0 for a
    /// failure.
    charged: AtomicU64,
}

impl Slot {
    fn new(state: SlotState, stamp: FileStamp) -> Self {
        Self {
            state: Mutex::new(state),
            ready: Condvar::new(),
            stamp,
            last_used: AtomicU64::new(0),
            charged: AtomicU64::new(0),
        }
    }

    fn lock(&self) -> MutexGuard<'_, SlotState> {
        // The lock is held only to read or set the state, never across a
        // decode, so nothing panics while it is held; poisoning is ignored.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The pyramid or the remembered failure, once made; `None` while the
    /// slot is empty or being made.
    fn made(&self) -> Option<Result<Arc<ImageU8Pyramid>, GetError>> {
        match &*self.lock() {
            SlotState::Made(value) => Some(value.clone()),
            SlotState::Empty | SlotState::Making => None,
        }
    }

    /// Set the state and wake every caller waiting on the slot.
    fn set(&self, state: SlotState) {
        *self.lock() = state;
        self.ready.notify_all();
    }
}

/// The decoding caller's hold on a slot it marked [`SlotState::Making`], with
/// its share of the build count. The share goes back before the slot changes
/// state, so a waiter that decodes next never finds this decode still
/// counted. A hold dropped without [`Self::finish`] (the caller was
/// cancelled, or it unwinds from a panic) gives the share back, empties the
/// slot and wakes its waiters, so no slot is left marked as being made by a
/// caller that has stopped, and the first waiter to take the lock decodes in
/// its place.
struct Making<'s, 'r> {
    slot: &'s Slot,
    share: Share<'r>,
    finished: bool,
}

impl Making<'_, '_> {
    fn finish(mut self, value: Result<Arc<ImageU8Pyramid>, GetError>) {
        self.finished = true;
        self.share.release();
        self.slot.set(SlotState::Made(value));
    }
}

impl Drop for Making<'_, '_> {
    fn drop(&mut self) {
        if !self.finished {
            self.share.release();
            self.slot.set(SlotState::Empty);
        }
    }
}

struct Inner {
    map: HashMap<PathBuf, Arc<Slot>>,
    /// The sum of every mapped slot's `charged`.
    bytes: u64,
}

impl Inner {
    fn remove(&mut self, path: &Path) {
        if let Some(old) = self.map.remove(path) {
            self.bytes -= old.charged.load(Ordering::Relaxed);
        }
    }
}

/// Decoded photographs as pyramids, keyed by path, shared across threads.
///
/// Every pyramid is `ImageU8::read_rgb` of the file followed by
/// [`ImageU8Pyramid::from_image`] at [`ImageU8Pyramid::full_levels`] of the
/// photograph's own size, so each entry's depth is set by its image. A
/// photograph with a side over [`MAX_JPEG_SIDE`] is refused from its header,
/// before any decode, as [`GetError::TooLarge`]. A file that cannot be read
/// or is refused is remembered as its error and costs nothing against the
/// budget. An entry records the file's length and modification time when it
/// is made; `get` and `get_many` compare them with the file on disk and decode
/// again when they differ, so a remembered failure is also retried once the
/// file appears or changes.
///
/// When a decode takes the bytes held over the budget, the least recently used
/// pyramids are dropped until the total is back under it, never the one just
/// decoded. Only the cache's `Arc` is dropped: a caller still holding a
/// pyramid keeps it alive.
///
/// Each path has one slot, empty, being made by one caller, or made. The
/// caller that finds it empty marks it as being made and decodes; the others
/// wait on it, looking at their own `Progress` every [`SLOT_WAIT_POLL`], and
/// stop with [`GetError::Cancelled`] when it is cancelled, leaving the decode
/// to its maker. A maker that is cancelled between the decode and building
/// the pyramid, or that unwinds, empties the slot, and a waiter decodes in its
/// place.
///
/// Every whole decode is first reserved on the cache's build count, an atomic
/// that belongs to this cache alone, against
/// [`PhotographCacheOptions::build_budget_bytes`]. Nothing waits on the
/// count: [`Self::reserve_builds`] grants at least one decode, so a caller
/// always makes progress, and [`Self::try_reserve_build`] is refused when its
/// decode does not fit.
pub struct PhotographCache {
    options: PhotographCacheOptions,
    inner: Mutex<Inner>,
    tick: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    decodes: AtomicU64,
    /// Estimated bytes of the whole decodes reserved now.
    build_reserved: AtomicU64,
    /// Called as a decode starts, once its header is read and the decode is
    /// reserved and before the pixels are read, and as a caller starts to
    /// wait on a slot being made: what lets a test hold a decode in progress,
    /// cancel a caller in the middle of a decode or a wait, make a decode
    /// panic, or read the build count at those moments.
    #[cfg(test)]
    on_slot_event: Option<Box<SlotHook>>,
}

/// The test build's hook on a cache's slot events, told the cache too so it
/// can read the build count.
#[cfg(test)]
type SlotHook = dyn Fn(&PhotographCache, SlotEvent) + Send + Sync;

/// What [`PhotographCache::on_slot_event`] is told.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotEvent {
    DecodeStarted,
    WaitStarted,
}

impl std::fmt::Debug for PhotographCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhotographCache")
            .field("stats", &self.stats())
            .finish()
    }
}

/// Whether a `get` decoded the photograph itself.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Found {
    Decoded,
    Reused,
}

/// One path's answer inside `get_many`.
type Answer = (Result<Arc<ImageU8Pyramid>, GetError>, Found);

impl PhotographCache {
    /// A cache holding up to `options.budget_bytes` of pyramids, each of
    /// [`ImageU8Pyramid::full_levels`] depth for its photograph's size, with
    /// the whole decodes in progress on it bounded by
    /// `options.build_budget_bytes`. A budget of 0 keeps nothing: every call
    /// decodes and hands back a pyramid no one else will see. A build budget
    /// of 0 lets each caller decode one photograph at a time.
    pub fn new(options: PhotographCacheOptions) -> Self {
        Self {
            options,
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                bytes: 0,
            }),
            tick: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            decodes: AtomicU64::new(0),
            build_reserved: AtomicU64::new(0),
            #[cfg(test)]
            on_slot_event: None,
        }
    }

    /// The budget the cache was made with. At 0 it keeps nothing, so a decode
    /// that one caller starts is never there for another to [`Self::peek`].
    pub fn budget_bytes(&self) -> u64 {
        self.options.budget_bytes
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
        let budget = self.options.build_budget_bytes;
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
        let before = update(&self.build_reserved, |reserved| {
            let bytes = bytes_per_build.saturating_mul(grant(reserved) as u64);
            Some(reserved.saturating_add(bytes))
        });
        // `update` always sets the count here, since the closure never
        // answers `None`; the grant is the one made from the count it found.
        let granted = before.map_or(0, grant);
        BuildReservation {
            count: &self.build_reserved,
            bytes_per_build,
            granted,
            released: AtomicU64::new(0),
        }
    }

    /// Reserve one whole decode of `bytes` estimated bytes only if it fits
    /// under this cache's build budget beside what other callers have
    /// reserved: `None` when it does not, so it never takes the count past the
    /// budget. Never waits; a caller that is refused asks again later.
    pub fn try_reserve_build(&self, bytes: u64) -> Option<BuildReservation<'_>> {
        let budget = self.options.build_budget_bytes;
        update(&self.build_reserved, |reserved| {
            reserved.checked_add(bytes).filter(|&after| after <= budget)
        })?;
        Some(BuildReservation {
            count: &self.build_reserved,
            bytes_per_build: bytes,
            granted: 1,
            released: AtomicU64::new(0),
        })
    }

    /// The pyramid for `path`, decoding on a miss. Waits while another thread
    /// is decoding the same path, and then returns its result rather than
    /// decoding a second time; if that thread is cancelled or unwinds, the
    /// first waiter decodes in its place. An error when the file does not
    /// exist, cannot be read, or has a side over [`MAX_JPEG_SIDE`]; that
    /// answer is remembered until `forget` or `clear`, or until the file's
    /// length or modification time changes.
    ///
    /// Polls `progress` for cancellation while it waits, every
    /// [`SLOT_WAIT_POLL`], and after its own decode, before it builds the
    /// pyramid, and answers [`GetError::Cancelled`] when it is cancelled,
    /// remembering nothing. It reports nothing through `progress`; a caller
    /// with nothing to cancel passes `&Progress::none()`.
    ///
    /// A decode it makes is reserved with [`Self::reserve_builds`] when it has
    /// marked the slot as being made, whether the slot was empty or a stopped
    /// maker has just emptied it, and given back when the decode ends, is
    /// cancelled or unwinds. It holds no share while it waits.
    ///
    /// A caller already inside a rayon parallel loop calls this rather than
    /// [`Self::get_many`].
    pub fn get(
        &self,
        path: &Path,
        progress: &Progress<'_>,
    ) -> Result<Arc<ImageU8Pyramid>, GetError> {
        self.get_found(path, progress, Share::None).0
    }

    /// The pyramid for `path` if it is decoded, without reading anything and
    /// without waiting on a decode in flight. What a panel calls every frame.
    ///
    /// It does not compare the entry with the file on disk, and counts as a
    /// use for the eviction order but not as a hit.
    pub fn peek(&self, path: &Path) -> Option<Arc<ImageU8Pyramid>> {
        match self.peek_state(path) {
            PeekedPhotograph::Decoded(pyramid) => Some(pyramid),
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
        let Some(slot) = self.lock().map.get(path).map(Arc::clone) else {
            return PeekedPhotograph::NotDecoded;
        };
        // The slot's lock is never held across a decode, so this does not
        // wait on one.
        let Some(value) = slot.made() else {
            return PeekedPhotograph::NotDecoded;
        };
        self.touch(&slot);
        match value {
            Ok(pyramid) => PeekedPhotograph::Decoded(pyramid),
            Err(_) => PeekedPhotograph::Unreadable,
        }
    }

    /// `get` for every path, the misses decoded in parallel on the rayon
    /// pool. Returns one entry per path, in order, and the tally the caller's
    /// phase note reports.
    ///
    /// It first looks for each path in the cache and reads the header of each
    /// one it does not find there made, to know its misses and their sizes.
    /// It then decodes the misses in groups, one parallel loop a group and the
    /// groups one after another, each group as many as
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
    /// that stays in the cache; a decode it stopped leaves its slot empty.
    // Spelled out rather than behind an alias, so the signature reads the same
    // here, in the docs and in the spec.
    #[allow(clippy::type_complexity)]
    pub fn get_many(
        &self,
        paths: &[&Path],
        progress: &Progress<'_>,
    ) -> Result<(Vec<Option<Arc<ImageU8Pyramid>>>, GetManyTally), Cancelled> {
        let total = paths.len();
        let answered = AtomicUsize::new(0);
        let answer = |found| {
            let n = answered.fetch_add(1, Ordering::Relaxed) + 1;
            progress.count(n as u64, Some(total as u64), "images");
            Some(found)
        };
        // A path made from the file as it is now is answered here; any other
        // is a miss, with the estimated bytes of its decode, 0 when its header
        // cannot be read (its `get` then stops at the header too, decoding
        // nothing).
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
        let pyramids = found
            .into_iter()
            .map(|entry| {
                // Every entry is `Some`, and none is `GetError::Cancelled`,
                // which only a cancelled `progress` gives, when the call was
                // not cancelled.
                let (pyramid, how) = entry.ok_or(Cancelled)?;
                match (&pyramid, how) {
                    (Err(_), _) => tally.unreadable += 1,
                    (Ok(_), Found::Decoded) => tally.read += 1,
                    (Ok(_), Found::Reused) => tally.reused += 1,
                }
                Ok(pyramid.ok())
            })
            .collect::<Result<Vec<_>, Cancelled>>()?;
        Ok((pyramids, tally))
    }

    /// Put `pyramid` in the cache as the decode of `path`, replacing any entry
    /// for it, and drop least recently used entries down to the budget as a
    /// decode would. A cache with a budget of 0 keeps nothing, so there this
    /// does nothing.
    ///
    /// For a caller whose pixels for `path` did not come from this cache's
    /// decode, such as a test fixture standing in for a photograph that is not
    /// on disk. The entry records the file's length and modification time as
    /// they are now (or that there is no file), so `get` answers with
    /// `pyramid` until the file changes. The pyramid should have
    /// [`ImageU8Pyramid::full_levels`] levels for its size, as a decoded one
    /// would.
    pub fn insert(&self, path: &Path, pyramid: Arc<ImageU8Pyramid>) {
        if self.options.budget_bytes == 0 {
            return;
        }
        let bytes = pyramid.byte_len() as u64;
        let slot = Arc::new(Slot::new(SlotState::Made(Ok(pyramid)), file_stamp(path)));
        {
            let mut inner = self.lock();
            inner.remove(path);
            inner.map.insert(path.to_path_buf(), Arc::clone(&slot));
        }
        self.touch(&slot);
        self.charge(path, &slot, bytes);
    }

    /// Drop one path's entry. The next `get` of it reads the file again.
    pub fn forget(&self, path: &Path) {
        self.lock().remove(path);
    }

    /// Drop every entry.
    pub fn clear(&self) {
        let mut inner = self.lock();
        inner.map.clear();
        inner.bytes = 0;
    }

    /// Entries, bytes held, the budget, the hit, miss and decode counters, the
    /// build budget and the bytes reserved on the build count.
    pub fn stats(&self) -> PhotographCacheStats {
        let inner = self.lock();
        let entries = inner
            .map
            .values()
            .filter(|slot| matches!(slot.made(), Some(Ok(_))))
            .count();
        PhotographCacheStats {
            entries,
            bytes: inner.bytes,
            budget_bytes: self.options.budget_bytes,
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            decodes: self.decodes.load(Ordering::Relaxed),
            build_budget_bytes: self.options.build_budget_bytes,
            build_reserved_bytes: self.build_reserved.load(Ordering::Acquire),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        // Nothing panics while the lock is held except on a broken invariant,
        // and the map is still usable then, so poisoning is ignored.
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[cfg(test)]
    fn slot_event(&self, event: SlotEvent) {
        if let Some(on_slot_event) = &self.on_slot_event {
            on_slot_event(self, event);
        }
    }

    fn touch(&self, slot: &Slot) {
        let tick = self.tick.fetch_add(1, Ordering::Relaxed);
        slot.last_used.store(tick, Ordering::Relaxed);
    }

    /// Decode `path` and build its pyramid, polling `progress` between the
    /// two: [`GetError::Cancelled`] when it was cancelled during the decode.
    /// Reads the header first and, when `share` holds nothing, reserves the
    /// decode with [`Self::reserve_builds`] before reading the pixels.
    ///
    /// Must not use rayon. It runs while its slot is marked as being made, and
    /// a rayon worker waiting in a nested join may steal another `get_many`
    /// item for the same path, which would find that slot being made by its
    /// own thread and wait on itself forever. Neither the file decode nor
    /// `ImageU8Pyramid::from_image` uses rayon's global pool. The decode may
    /// start threads of its own (one per colour component in `jpeg-decoder`,
    /// a private pool in `exr`), which take no work from the global pool; see
    /// the photograph-cache spec.
    fn decode<'r>(
        &'r self,
        path: &Path,
        progress: &Progress<'_>,
        share: &mut Share<'r>,
    ) -> Result<Arc<ImageU8Pyramid>, GetError> {
        let unreadable = |e: ::image::ImageError| match e {
            ::image::ImageError::IoError(io) if io.kind() == std::io::ErrorKind::NotFound => {
                GetError::NotFound(path.to_path_buf())
            }
            e => GetError::Unreadable(format!("{}: {e}", path.display())),
        };
        // The header decides the refusal, so a photograph too large for the
        // pyramid readers is never decoded.
        let (width, height) = image_dimensions(path).map_err(unreadable)?;
        if width > MAX_JPEG_SIDE || height > MAX_JPEG_SIDE {
            return Err(GetError::TooLarge {
                path: path.to_path_buf(),
                width,
                height,
            });
        }
        if let Share::None = share {
            *share = Share::Own(self.reserve_builds(whole_decode_bytes(width, height), 1));
        }
        #[cfg(test)]
        self.slot_event(SlotEvent::DecodeStarted);
        let image = ImageU8::read_rgb(path).map_err(unreadable)?;
        self.decodes.fetch_add(1, Ordering::Relaxed);
        if progress.is_cancelled() {
            return Err(GetError::Cancelled);
        }
        let levels = ImageU8Pyramid::full_levels(image.width, image.height);
        Ok(Arc::new(ImageU8Pyramid::from_image(image, levels)))
    }

    /// The value of `path`'s slot when it is made from the file as it is now,
    /// counted as a hit; `None` otherwise. Reads nothing but the file's stamp
    /// and never waits on a decode.
    fn made_now(&self, path: &Path) -> Option<Result<Arc<ImageU8Pyramid>, GetError>> {
        if self.options.budget_bytes == 0 {
            return None;
        }
        let stamp = file_stamp(path);
        let slot = match self.lock().map.get(path) {
            Some(slot) if slot.stamp == stamp => Arc::clone(slot),
            _ => return None,
        };
        let value = slot.made()?;
        self.touch(&slot);
        self.hits.fetch_add(1, Ordering::Relaxed);
        Some(value)
    }

    /// `get`, telling whether it decoded. `share` is the caller's share of
    /// the build count when it already holds one (a `get_many` member's), or
    /// [`Share::None`] for a `get`, which reserves when it decodes. Whatever
    /// the outcome, the share has been given back when this returns.
    fn get_found<'r>(
        &'r self,
        path: &Path,
        progress: &Progress<'_>,
        mut share: Share<'r>,
    ) -> (Result<Arc<ImageU8Pyramid>, GetError>, Found) {
        if self.options.budget_bytes == 0 {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return (self.decode(path, progress, &mut share), Found::Decoded);
        }
        // Stat with the lock released; it is a system call.
        let stamp = file_stamp(path);
        let slot = {
            let mut inner = self.lock();
            match inner.map.get(path) {
                Some(slot) if slot.stamp == stamp => Arc::clone(slot),
                _ => {
                    inner.remove(path);
                    let slot = Arc::new(Slot::new(SlotState::Empty, stamp));
                    inner.map.insert(path.to_path_buf(), Arc::clone(&slot));
                    slot
                }
            }
        };
        self.touch(&slot);

        // Wait while another caller makes the slot, and take it over when it
        // is empty, as it is before anyone has decoded it and after its maker
        // stopped.
        #[cfg(test)]
        let mut told_waiting = false;
        let mut making = {
            let mut state = slot.lock();
            loop {
                match &*state {
                    SlotState::Made(value) => {
                        self.hits.fetch_add(1, Ordering::Relaxed);
                        return (value.clone(), Found::Reused);
                    }
                    SlotState::Making => {
                        if progress.is_cancelled() {
                            return (Err(GetError::Cancelled), Found::Reused);
                        }
                        #[cfg(test)]
                        if !std::mem::replace(&mut told_waiting, true) {
                            self.slot_event(SlotEvent::WaitStarted);
                        }
                        state = slot
                            .ready
                            .wait_timeout(state, SLOT_WAIT_POLL)
                            .unwrap_or_else(|e| e.into_inner())
                            .0;
                    }
                    SlotState::Empty => {
                        *state = SlotState::Making;
                        break Making {
                            slot: &slot,
                            share,
                            finished: false,
                        };
                    }
                }
            }
        };
        self.misses.fetch_add(1, Ordering::Relaxed);
        let value = self.decode(path, progress, &mut making.share);
        if let Err(GetError::Cancelled) = value {
            // Dropping the hold empties the slot and wakes its waiters.
            drop(making);
            return (value, Found::Decoded);
        }
        making.finish(value.clone());
        if let Ok(pyramid) = &value {
            self.charge(path, &slot, pyramid.byte_len() as u64);
        }
        (value, Found::Decoded)
    }

    /// Count `bytes` for the slot just decoded into, and evict down to the
    /// budget. Nothing is counted when the slot left the map while it was
    /// decoding (forgotten, cleared, evicted or replaced by a newer file):
    /// its caller still gets the pyramid, but the cache does not keep it.
    fn charge(&self, path: &Path, slot: &Arc<Slot>, bytes: u64) {
        let mut inner = self.lock();
        let mapped = inner
            .map
            .get(path)
            .is_some_and(|mapped| Arc::ptr_eq(mapped, slot));
        if !mapped {
            return;
        }
        slot.charged.store(bytes, Ordering::Relaxed);
        inner.bytes += bytes;
        if inner.bytes <= self.options.budget_bytes {
            return;
        }
        // Oldest first. Slots that hold no bytes (failures, decodes in flight)
        // would free nothing and are left alone.
        let mut candidates: Vec<(u64, PathBuf)> = inner
            .map
            .iter()
            .filter(|(_, other)| {
                !Arc::ptr_eq(other, slot) && other.charged.load(Ordering::Relaxed) > 0
            })
            .map(|(p, other)| (other.last_used.load(Ordering::Relaxed), p.clone()))
            .collect();
        candidates.sort_unstable_by_key(|(tick, _)| *tick);
        for (_, victim) in candidates {
            if inner.bytes <= self.options.budget_bytes {
                break;
            }
            inner.remove(&victim);
        }
    }
}

#[cfg(test)]
mod tests;
