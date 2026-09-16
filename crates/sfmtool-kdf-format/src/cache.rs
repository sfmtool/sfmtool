// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use xxhash_rust::xxh3::Xxh3DefaultBuilder;

use crate::{DecodedTreeChunk, FeatureGeometry, FeatureOrigin, KdfError, KdfIoStats, KdfScalar};

/// Dense internal keys use XXH3, already a dependency for section integrity.
/// Callers cannot supply arbitrary cache keys.
type KeyMap<V> = HashMap<CacheKey, V, Xxh3DefaultBuilder>;
type KeySet = HashSet<CacheKey, Xxh3DefaultBuilder>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CacheKey {
    Tree(u32, u32),
    Descriptor(u32),
    Geometry(u32),
    Origin(u32),
}

pub(crate) enum Cached<S: KdfScalar> {
    Tree(DecodedTreeChunk<S>),
    Descriptor(Vec<S>),
    Geometry(Vec<FeatureGeometry>),
    Origin(Vec<FeatureOrigin>),
}

impl<S: KdfScalar> Cached<S> {
    pub(crate) fn bytes(&self) -> usize {
        match self {
            Self::Tree(v) => v.decoded_bytes,
            Self::Descriptor(v) => std::mem::size_of_val(v.as_slice()),
            Self::Geometry(v) => std::mem::size_of_val(v.as_slice()),
            Self::Origin(v) => std::mem::size_of_val(v.as_slice()),
        }
    }
}

struct Entry<S: KdfScalar> {
    value: Arc<Cached<S>>,
    bytes: usize,
    recency: usize,
}

/// Stable slots avoid hashing neighbouring keys when promoting an entry.
struct Recency {
    key: CacheKey,
    prev: Option<usize>,
    next: Option<usize>,
}

#[derive(Default)]
struct Counters {
    read_calls: u64,
    compressed_bytes: u64,
    decoded_bytes: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
    waits: u64,
    peak_resident: usize,
}

struct State<S: KdfScalar> {
    entries: KeyMap<Entry<S>>,
    recency: Vec<Recency>,
    free_recency: Vec<usize>,
    oldest: Option<usize>,
    newest: Option<usize>,
    loading: KeySet,
    reserved: usize,
    resident: usize,
    counters: Counters,
}

/// One independently locked slice of the cache.
///
/// Residency and eviction are per shard. Cache hits touch no global counter;
/// the separate global decode gate is used only when an entry must be loaded.
struct Shard<S: KdfScalar> {
    state: Mutex<State<S>>,
    changed: Condvar,
    /// Admissions currently blocked on this shard's capacity.
    ///
    /// Read without the lock by a dropping pin, which is why it is an atomic and
    /// not a `State` field. The ordering that makes this safe: a pin releases its
    /// hold, then loads this; a waiter increments it under the lock, then checks
    /// capacity. In the sequentially consistent total order either the pin sees
    /// the waiter and wakes it, or the waiter's capacity check follows the pin's
    /// release and finds the room it needed.
    waiters: AtomicUsize,
    capacity: usize,
}

pub(crate) struct Cache<S: KdfScalar> {
    /// A power-of-two number of shards, so selection is a mask.
    shards: Vec<Shard<S>>,
    shard_mask: usize,
    capacity: usize,
    in_flight_capacity: usize,
    address_map_bytes: usize,
    decode_bytes: Mutex<(usize, usize)>,
    decode_changed: Condvar,
    decode_waiters: AtomicUsize,
}

/// A cache pin. Dropping it releases the hold and, if an admission is waiting on
/// capacity, wakes it — so pinned residency counts against the byte limit
/// instead of being silently evicted from accounting.
pub(crate) struct CachePin<'a, S: KdfScalar> {
    /// `Option` so [`Drop`] can release the hold *before* checking for waiters.
    /// Field drop runs after the `drop` body, which would otherwise mean waking
    /// a waiter that still sees this entry pinned.
    value: Option<Arc<Cached<S>>>,
    owner: &'a Cache<S>,
    /// Which shard to wake. Pins are taken and dropped far more often than any
    /// shard is contended, so this is carried rather than recomputed.
    shard: usize,
}

impl<S: KdfScalar> std::ops::Deref for CachePin<'_, S> {
    type Target = Cached<S>;
    fn deref(&self) -> &Self::Target {
        self.value
            .as_ref()
            .expect("pin holds its value until dropped")
    }
}

impl<S: KdfScalar> Drop for CachePin<'_, S> {
    fn drop(&mut self) {
        // A search takes and drops one pin per node and per descriptor it
        // examines, so this runs hundreds of times per query. Notifying
        // unconditionally made every one of those a condition-variable wake with
        // no waiter to receive it.
        drop(self.value.take());
        {
            let shard = &self.owner.shards[self.shard];
            if shard.waiters.load(Ordering::SeqCst) > 0 {
                // Synchronize with the predicate check and atomic unlock in
                // Condvar::wait; notification alone can precede the actual wait.
                let _guard = shard.state.lock().unwrap();
                shard.changed.notify_all();
            }
        }
    }
}

/// Shards to split the cache into, at most this many.
///
/// This caps budget fragmentation; the largest validated item further limits
/// the count so each shard can admit every item mapped to it.
const MAX_SHARDS: usize = 16;

impl<S: KdfScalar> Cache<S> {
    /// Build a cache whose shards can each hold at least one `largest_item`.
    ///
    /// The shard count is bounded by `capacity / largest_item`, not chosen freely:
    /// admission is per shard, so a shard smaller than the biggest item a caller
    /// may ask for could never admit it and the caller would wait forever. One
    /// shard is always valid, which is what a cache only just large enough for a
    /// single item collapses to.
    pub(crate) fn new(
        capacity: usize,
        in_flight_capacity: usize,
        address_map_bytes: usize,
        largest_item: usize,
    ) -> Arc<Self> {
        let affordable = capacity / largest_item.max(1);
        let mut shards = 1usize;
        while shards * 2 <= affordable.min(MAX_SHARDS) {
            shards *= 2;
        }
        Arc::new(Self {
            shards: (0..shards)
                .map(|_| Shard {
                    state: Mutex::new(State {
                        entries: KeyMap::default(),
                        recency: Vec::new(),
                        free_recency: Vec::new(),
                        oldest: None,
                        newest: None,
                        loading: KeySet::default(),
                        reserved: 0,
                        resident: 0,
                        counters: Counters::default(),
                    }),
                    changed: Condvar::new(),
                    waiters: AtomicUsize::new(0),
                    capacity: capacity / shards,
                })
                .collect(),
            shard_mask: shards - 1,
            capacity,
            in_flight_capacity,
            address_map_bytes,
            decode_bytes: Mutex::new((0, 0)),
            decode_changed: Condvar::new(),
            decode_waiters: AtomicUsize::new(0),
        })
    }

    /// Which shard owns a key.
    ///
    /// Keys are dense small integers — block and chunk indexes — so the low bits
    /// already spread uniformly and no hashing is needed. Sharding on the cached
    /// *contents* instead, say a descriptor's leading bytes, would be badly
    /// skewed: SIFT descriptors carry many small and zero components, so most
    /// keys would land in a few shards.
    fn shard_index_of(&self, key: CacheKey) -> usize {
        let spread = match key {
            CacheKey::Tree(tree, chunk) => (chunk as usize) ^ (tree as usize).wrapping_mul(0x9e37),
            CacheKey::Descriptor(block) => block as usize,
            CacheKey::Geometry(block) => (block as usize).wrapping_mul(0x85eb_ca6b),
            CacheKey::Origin(block) => block as usize,
        };
        spread & self.shard_mask
    }

    fn unlink(state: &mut State<S>, slot: usize) {
        let link = &state.recency[slot];
        let (prev, next) = (link.prev, link.next);
        if let Some(prev) = prev {
            state.recency[prev].next = next;
        } else {
            state.oldest = next;
        }
        if let Some(next) = next {
            state.recency[next].prev = prev;
        } else {
            state.newest = prev;
        }
    }

    fn append(state: &mut State<S>, slot: usize) {
        state.recency[slot].prev = state.newest;
        state.recency[slot].next = None;
        if let Some(last) = state.newest {
            state.recency[last].next = Some(slot);
        } else {
            state.oldest = Some(slot);
        }
        state.newest = Some(slot);
    }

    /// Hits and removal are O(1); eviction visits only older pinned entries
    /// before its victim, rather than scanning every resident hash bucket.
    fn touch(state: &mut State<S>, slot: usize) {
        if state.newest != Some(slot) {
            Self::unlink(state, slot);
            Self::append(state, slot);
        }
    }

    fn evict_unpinned(shard: &Shard<S>, state: &mut State<S>, needed: usize) {
        let mut candidate = state.oldest;
        while state
            .resident
            .saturating_add(state.reserved)
            .saturating_add(needed)
            > shard.capacity
        {
            let Some(slot) = candidate else { break };
            candidate = state.recency[slot].next;
            let key = state.recency[slot].key;
            if Arc::strong_count(&state.entries[&key].value) != 1 {
                continue;
            }
            let entry = state.entries.remove(&key).expect("recency entry exists");
            Self::unlink(state, slot);
            state.free_recency.push(slot);
            state.resident -= entry.bytes;
            state.counters.evictions += 1;
        }
    }

    pub(crate) fn get_or_load<F>(
        self: &Arc<Self>,
        key: CacheKey,
        declared: usize,
        load: F,
    ) -> Result<CachePin<'_, S>, KdfError>
    where
        F: FnOnce() -> Result<(Cached<S>, u64), KdfError>,
    {
        if declared > self.capacity || declared > self.in_flight_capacity {
            return Err(KdfError::ResourceLimit(format!(
                "decoded item requires {declared} bytes; cache={} in-flight={}",
                self.capacity, self.in_flight_capacity
            )));
        }
        let shard_index = self.shard_index_of(key);
        let shard = &self.shards[shard_index];
        if declared > shard.capacity {
            return Err(KdfError::ResourceLimit(format!(
                "decoded item requires {declared} bytes; this cache splits {} bytes                  into {} shards of {}",
                self.capacity,
                self.shards.len(),
                shard.capacity
            )));
        }
        let mut load = Some(load);
        loop {
            let mut state = shard.state.lock().unwrap();
            if let Some(entry) = state.entries.get(&key) {
                let value = Arc::clone(&entry.value);
                let slot = entry.recency;
                Self::touch(&mut state, slot);
                state.counters.hits += 1;
                return Ok(CachePin {
                    value: Some(value),
                    owner: self,
                    shard: shard_index,
                });
            }
            if state.loading.contains(&key) {
                state.counters.waits += 1;
                shard.waiters.fetch_add(1, Ordering::SeqCst);
                let woken = shard.changed.wait(state);
                shard.waiters.fetch_sub(1, Ordering::SeqCst);
                drop(woken.unwrap());
                continue;
            }
            shard.waiters.fetch_add(1, Ordering::SeqCst);
            Self::evict_unpinned(shard, &mut state, declared);
            if state.resident + state.reserved + declared > shard.capacity {
                let woken = shard.changed.wait(state);
                shard.waiters.fetch_sub(1, Ordering::SeqCst);
                drop(woken.unwrap());
                continue;
            }
            shard.waiters.fetch_sub(1, Ordering::SeqCst);
            state.loading.insert(key);
            state.reserved += declared;
            state.counters.misses += 1;
            drop(state);

            // Global decode admission is only a miss-path operation. Dividing
            // this limit by shards either disables useful sharding or silently
            // multiplies the promised limit when each share is rounded up.
            let mut decoding = self.decode_bytes.lock().unwrap();
            while decoding.0 + declared > self.in_flight_capacity {
                self.decode_waiters.fetch_add(1, Ordering::Relaxed);
                decoding = self.decode_changed.wait(decoding).unwrap();
                self.decode_waiters.fetch_sub(1, Ordering::Relaxed);
            }
            decoding.0 += declared;
            decoding.1 = decoding.1.max(decoding.0);
            drop(decoding);
            let result = load.take().expect("loader runs once")();
            let mut state = shard.state.lock().unwrap();
            state.loading.remove(&key);
            state.reserved -= declared;
            {
                let mut decoding = self.decode_bytes.lock().unwrap();
                decoding.0 -= declared;
                if self.decode_waiters.load(Ordering::Relaxed) != 0 {
                    self.decode_changed.notify_all();
                }
            }
            match result {
                Ok((value, compressed_bytes)) => {
                    let actual = value.bytes();
                    if actual != declared {
                        if shard.waiters.load(Ordering::SeqCst) != 0 {
                            shard.changed.notify_all();
                        }
                        return Err(KdfError::ShapeMismatch(format!(
                            "decoded item declared {declared} bytes but produced {actual}"
                        )));
                    }
                    let value = Arc::new(value);
                    state.resident += actual;
                    state.counters.read_calls += 1;
                    state.counters.compressed_bytes += compressed_bytes;
                    state.counters.decoded_bytes += actual as u64;
                    state.counters.peak_resident = state.counters.peak_resident.max(state.resident);
                    let link = Recency {
                        key,
                        prev: None,
                        next: None,
                    };
                    let slot = if let Some(slot) = state.free_recency.pop() {
                        state.recency[slot] = link;
                        slot
                    } else {
                        state.recency.push(link);
                        state.recency.len() - 1
                    };
                    Self::append(&mut state, slot);
                    state.entries.insert(
                        key,
                        Entry {
                            value: Arc::clone(&value),
                            bytes: actual,
                            recency: slot,
                        },
                    );
                    if shard.waiters.load(Ordering::SeqCst) != 0 {
                        shard.changed.notify_all();
                    }
                    return Ok(CachePin {
                        value: Some(value),
                        owner: self,
                        shard: shard_index,
                    });
                }
                Err(e) => {
                    if shard.waiters.load(Ordering::SeqCst) != 0 {
                        shard.changed.notify_all();
                    }
                    return Err(e);
                }
            }
        }
    }

    /// Zero the cumulative counters, keeping the live gauges.
    ///
    /// Resident and in-flight bytes describe what the cache is holding right
    /// now, so they survive; the peaks restart from those current values rather
    /// than from zero, which would claim a peak below a byte count already
    /// resident. A benchmark uses this to separate an open from the queries
    /// that follow it, or a cold pass from a warm one, without reopening.
    pub(crate) fn reset_counters(&self) {
        {
            let mut decoding = self.decode_bytes.lock().unwrap();
            decoding.1 = decoding.0;
        }
        for shard in &self.shards {
            let mut s = shard.state.lock().unwrap();
            let resident = s.resident;
            s.counters = Counters {
                peak_resident: resident,
                ..Counters::default()
            };
        }
    }

    /// Counters summed over every shard.
    ///
    /// The byte gauges add up because shards hold disjoint entries. The peaks are
    /// summed too, which overstates a true instantaneous peak — shards need not
    /// have peaked together — but it is the bound the budget is set against, and
    /// a maximum over shards would understate it badly.
    pub(crate) fn stats(&self) -> KdfIoStats {
        let mut out = KdfIoStats {
            read_calls: 0,
            compressed_bytes: 0,
            decoded_bytes: 0,
            cache_hits: 0,
            cache_misses: 0,
            evictions: 0,
            duplicate_load_waits: 0,
            resident_bytes: 0,
            peak_resident_bytes: 0,
            in_flight_bytes: 0,
            peak_in_flight_bytes: 0,
            address_map_bytes: self.address_map_bytes,
        };
        for shard in &self.shards {
            let s = shard.state.lock().unwrap();
            out.read_calls += s.counters.read_calls;
            out.compressed_bytes += s.counters.compressed_bytes;
            out.decoded_bytes += s.counters.decoded_bytes;
            out.cache_hits += s.counters.hits;
            out.cache_misses += s.counters.misses;
            out.evictions += s.counters.evictions;
            out.duplicate_load_waits += s.counters.waits;
            out.resident_bytes += s.resident;
            out.peak_resident_bytes += s.counters.peak_resident;
        }
        let decoding = self.decode_bytes.lock().unwrap();
        out.in_flight_bytes = decoding.0;
        out.peak_in_flight_bytes = decoding.1;
        out
    }
}

#[cfg(test)]
mod tests;
