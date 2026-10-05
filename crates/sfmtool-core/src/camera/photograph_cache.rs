// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Decoded photographs as image pyramids, keyed by file path, shared across
//! threads and bounded by a byte budget.
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
//! [`PhotographCache::get`] blocks while another thread decodes the same path;
//! [`PhotographCache::peek`] never reads a file and never waits, which is what a
//! caller that draws every frame wants. The design is in the spec
//! `specs/core/camera/photograph-cache.md`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::SystemTime;

use rayon::prelude::*;

use crate::camera::image::{ImageU8, ImageU8Pyramid};
use crate::progress::{Cancelled, Progress};

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
/// Physical memory is read with `GlobalMemoryStatusEx` on Windows and with
/// `sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGESIZE)` on Linux and macOS. Other
/// targets take the 4 GiB fallback.
pub fn default_budget_bytes() -> u64 {
    let env = std::env::var(BUDGET_ENV_VAR).ok();
    budget_from(env.as_deref(), physical_memory_bytes())
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

#[cfg(windows)]
fn physical_memory_bytes() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: MEMORYSTATUSEX is plain data, so all zeroes is a valid value, and
    // GlobalMemoryStatusEx only writes into the struct it is handed, whose
    // `dwLength` it requires to be set to its size.
    unsafe {
        let mut status: MEMORYSTATUSEX = std::mem::zeroed();
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        if GlobalMemoryStatusEx(&mut status) != 0 && status.ullTotalPhys > 0 {
            Some(status.ullTotalPhys)
        } else {
            None
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn physical_memory_bytes() -> Option<u64> {
    // SAFETY: sysconf takes an integer name and returns an integer; it touches
    // no memory of ours.
    let (pages, page_size) = unsafe {
        (
            libc::sysconf(libc::_SC_PHYS_PAGES),
            libc::sysconf(libc::_SC_PAGESIZE),
        )
    };
    if pages > 0 && page_size > 0 {
        (pages as u64).checked_mul(page_size as u64)
    } else {
        None
    }
}

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
fn physical_memory_bytes() -> Option<u64> {
    None
}

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
    /// `get_many` counts as one `get`. `peek` is not counted.
    pub hits: u64,
    /// Decodes run, whether they succeeded or not.
    pub misses: u64,
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

/// One path's entry. Shared between the map and every thread using it, so a
/// thread can decode into it with the map lock released.
struct Slot {
    value: OnceLock<Option<Arc<ImageU8Pyramid>>>,
    stamp: FileStamp,
    /// The cache's tick when the slot was last asked for, for LRU order.
    last_used: AtomicU64,
    /// Bytes this slot added to [`Inner::bytes`]. Read and written only under
    /// the map lock; 0 until the decoding thread charges it, and 0 for a
    /// failure.
    charged: AtomicU64,
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
/// [`ImageU8Pyramid::from_image`] with the cache's level count. A file that
/// cannot be read is remembered as `None` and costs nothing against the
/// budget. An entry records the file's length and modification time when it
/// is made; `get` and `get_many` compare them with the file on disk and decode
/// again when they differ, so a remembered failure is also retried once the
/// file appears or changes.
///
/// When a decode takes the bytes held over the budget, the least recently used
/// pyramids are dropped until the total is back under it, never the one just
/// decoded. Only the cache's `Arc` is dropped: a caller still holding a
/// pyramid keeps it alive.
pub struct PhotographCache {
    budget_bytes: u64,
    levels: usize,
    inner: Mutex<Inner>,
    tick: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl std::fmt::Debug for PhotographCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhotographCache")
            .field("levels", &self.levels)
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

impl PhotographCache {
    /// A cache holding up to `budget_bytes` of pyramids of `levels` levels.
    /// A budget of 0 keeps nothing: every call decodes and hands back a
    /// pyramid no one else will see.
    ///
    /// # Panics
    ///
    /// When `levels` is 0.
    pub fn new(budget_bytes: u64, levels: usize) -> Self {
        assert!(levels >= 1, "PhotographCache: a pyramid needs a level");
        Self {
            budget_bytes,
            levels,
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                bytes: 0,
            }),
            tick: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// The pyramid depth every entry is built with.
    pub fn levels(&self) -> usize {
        self.levels
    }

    /// The budget the cache was made with. At 0 it keeps nothing, so a decode
    /// that one caller starts is never there for another to [`Self::peek`].
    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    /// The pyramid for `path`, decoding on a miss. Blocks while another thread
    /// is decoding the same path, and then returns its result rather than
    /// decoding a second time. `None` when the file cannot be read; that
    /// answer is remembered until `forget` or `clear`, or until the file's
    /// length or modification time changes.
    ///
    /// A caller already inside a rayon parallel loop calls this rather than
    /// [`Self::get_many`].
    pub fn get(&self, path: &Path) -> Option<Arc<ImageU8Pyramid>> {
        self.get_found(path).0
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
        let Some(value) = slot.value.get() else {
            return PeekedPhotograph::NotDecoded;
        };
        self.touch(&slot);
        match value {
            Some(pyramid) => PeekedPhotograph::Decoded(Arc::clone(pyramid)),
            None => PeekedPhotograph::Unreadable,
        }
    }

    /// `get` for every path, the misses decoded in parallel on the rayon
    /// pool. Returns one entry per path, in order, and the tally the caller's
    /// phase note reports.
    ///
    /// Counts `images` through `progress` as each path is answered and polls
    /// it for cancellation before each decode. It opens no phase of its own:
    /// the caller names the phase (the viewer's is `decode images`) and writes
    /// its note from the tally, and may add [`Self::stats`] to it.
    ///
    /// # Errors
    ///
    /// [`Cancelled`] when `progress` was cancelled. What was decoded before
    /// that stays in the cache.
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
        let found: Vec<Option<(Option<Arc<ImageU8Pyramid>>, Found)>> = paths
            .par_iter()
            .map(|path| {
                if progress.is_cancelled() {
                    return None;
                }
                let found = self.get_found(path);
                let n = answered.fetch_add(1, Ordering::Relaxed) + 1;
                progress.count(n as u64, Some(total as u64), "images");
                Some(found)
            })
            .collect();
        progress.check_cancel()?;
        let mut tally = GetManyTally::default();
        let pyramids = found
            .into_iter()
            .map(|entry| {
                // Every entry is `Some` when the call was not cancelled.
                let (pyramid, how) = entry.ok_or(Cancelled)?;
                match (&pyramid, how) {
                    (None, _) => tally.unreadable += 1,
                    (Some(_), Found::Decoded) => tally.read += 1,
                    (Some(_), Found::Reused) => tally.reused += 1,
                }
                Ok(pyramid)
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
    /// [`Self::levels`] levels, as a decoded one would.
    pub fn insert(&self, path: &Path, pyramid: Arc<ImageU8Pyramid>) {
        if self.budget_bytes == 0 {
            return;
        }
        let bytes = pyramid.byte_len() as u64;
        let slot = Arc::new(Slot {
            value: OnceLock::from(Some(pyramid)),
            stamp: file_stamp(path),
            last_used: AtomicU64::new(0),
            charged: AtomicU64::new(0),
        });
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

    /// Entries, bytes held, the budget, and the hit and miss counters.
    pub fn stats(&self) -> PhotographCacheStats {
        let inner = self.lock();
        let entries = inner
            .map
            .values()
            .filter(|slot| matches!(slot.value.get(), Some(Some(_))))
            .count();
        PhotographCacheStats {
            entries,
            bytes: inner.bytes,
            budget_bytes: self.budget_bytes,
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        // Nothing panics while the lock is held except on a broken invariant,
        // and the map is still usable then, so poisoning is ignored.
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn touch(&self, slot: &Slot) {
        let tick = self.tick.fetch_add(1, Ordering::Relaxed);
        slot.last_used.store(tick, Ordering::Relaxed);
    }

    /// Must not use rayon. It runs inside a slot's `get_or_init`, and a rayon
    /// worker waiting in a nested join may steal another `get_many` item for
    /// the same path, which would re-enter that `OnceLock` on the same thread
    /// and deadlock. The file decode and `ImageU8Pyramid::from_image` are
    /// serial.
    fn decode(&self, path: &Path) -> Option<Arc<ImageU8Pyramid>> {
        let image = ImageU8::read_rgb(path).ok()?;
        Some(Arc::new(ImageU8Pyramid::from_image(image, self.levels)))
    }

    fn get_found(&self, path: &Path) -> (Option<Arc<ImageU8Pyramid>>, Found) {
        if self.budget_bytes == 0 {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return (self.decode(path), Found::Decoded);
        }
        // Stat with the lock released; it is a system call.
        let stamp = file_stamp(path);
        let slot = {
            let mut inner = self.lock();
            match inner.map.get(path) {
                Some(slot) if slot.stamp == stamp => Arc::clone(slot),
                _ => {
                    inner.remove(path);
                    let slot = Arc::new(Slot {
                        value: OnceLock::new(),
                        stamp,
                        last_used: AtomicU64::new(0),
                        charged: AtomicU64::new(0),
                    });
                    inner.map.insert(path.to_path_buf(), Arc::clone(&slot));
                    slot
                }
            }
        };
        self.touch(&slot);

        let mut decoded_here = false;
        let value = slot
            .value
            .get_or_init(|| {
                decoded_here = true;
                self.decode(path)
            })
            .clone();
        if !decoded_here {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return (value, Found::Reused);
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        if let Some(pyramid) = &value {
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
        if inner.bytes <= self.budget_bytes {
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
            if inner.bytes <= self.budget_bytes {
                break;
            }
            inner.remove(&victim);
        }
    }
}

#[cfg(test)]
mod tests;
