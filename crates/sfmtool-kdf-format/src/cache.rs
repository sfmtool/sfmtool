// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Condvar, Mutex, Weak};

use crate::{DecodedTreeChunk, FeatureOrigin, KdfError, KdfIoStats, KdfScalar};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CacheKey {
    Tree(u32, u32),
    Descriptor(u32),
    Origin(u32),
}

pub(crate) enum Cached<S: KdfScalar> {
    Tree(DecodedTreeChunk<S>),
    Descriptor(Vec<S>),
    Origin(Vec<FeatureOrigin>),
}

impl<S: KdfScalar> Cached<S> {
    pub(crate) fn bytes(&self) -> usize {
        match self {
            Self::Tree(v) => v.decoded_bytes,
            Self::Descriptor(v) => std::mem::size_of_val(v.as_slice()),
            Self::Origin(v) => std::mem::size_of_val(v.as_slice()),
        }
    }
}

struct Entry<S: KdfScalar> {
    value: Arc<Cached<S>>,
    bytes: usize,
    /// Recency stamp, from [`State::clock`]. Ordering these is what picks an
    /// eviction victim; the absolute values mean nothing.
    last_used: u64,
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
    peak_in_flight: usize,
}

struct State<S: KdfScalar> {
    entries: HashMap<CacheKey, Entry<S>>,
    /// Monotonic source of recency stamps. A `u64` at one tick per cache access
    /// cannot wrap in any run this format will see.
    clock: u64,
    loading: HashSet<CacheKey>,
    reserved: usize,
    resident: usize,
    in_flight: usize,
    counters: Counters,
}

pub(crate) struct Cache<S: KdfScalar> {
    state: Mutex<State<S>>,
    changed: Condvar,
    capacity: usize,
    in_flight_capacity: usize,
    address_map_bytes: usize,
}

/// A cache pin. Dropping it wakes admission waiters so pinned residency is
/// included in the byte limit instead of being silently evicted from accounting.
pub(crate) struct CachePin<S: KdfScalar> {
    value: Arc<Cached<S>>,
    owner: Weak<Cache<S>>,
}

impl<S: KdfScalar> std::ops::Deref for CachePin<S> {
    type Target = Cached<S>;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<S: KdfScalar> Drop for CachePin<S> {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.upgrade() {
            owner.changed.notify_all();
        }
    }
}

impl<S: KdfScalar> Cache<S> {
    pub(crate) fn new(
        capacity: usize,
        in_flight_capacity: usize,
        address_map_bytes: usize,
    ) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State {
                entries: HashMap::new(),
                clock: 0,
                loading: HashSet::new(),
                reserved: 0,
                resident: 0,
                in_flight: 0,
                counters: Counters::default(),
            }),
            changed: Condvar::new(),
            capacity,
            in_flight_capacity,
            address_map_bytes,
        })
    }

    /// Mark an entry as most recently used, in constant time.
    ///
    /// This runs on **every cache hit**, so its cost is the cache's per-access
    /// cost. An ordered recency list makes that O(resident entries) — promoting
    /// an entry means finding it and removing it from the middle — which is
    /// invisible on a small cache and dominant on a large one: a file holding
    /// ~23,000 descriptor blocks measured 33.5 us per fully-cached access, three
    /// orders of magnitude above the hash lookup it should have been.
    ///
    /// Stamping instead moves that work to eviction, which has to inspect the
    /// same set anyway to skip pinned entries. The trade is sound because hits
    /// vastly outnumber evictions: in a resident working set there are no
    /// evictions at all, and even the most cache-starved configuration measured
    /// ran three hits per eviction.
    fn touch(state: &mut State<S>, key: CacheKey) {
        state.clock += 1;
        let stamp = state.clock;
        if let Some(entry) = state.entries.get_mut(&key) {
            entry.last_used = stamp;
        }
    }

    /// Evict least-recently-used unpinned entries until `needed` bytes fit.
    ///
    /// Each pass scans for the oldest stamp among unpinned entries. A pinned
    /// entry is one a caller still holds a [`CachePin`] for, so it cannot be
    /// dropped without invalidating a borrow; the loop stops when every
    /// remaining entry is pinned, which is the caller's cue to wait for one to
    /// be released rather than to spin.
    fn evict_unpinned(&self, state: &mut State<S>, needed: usize) {
        while state
            .resident
            .saturating_add(state.reserved)
            .saturating_add(needed)
            > self.capacity
        {
            let victim = state
                .entries
                .iter()
                .filter(|(_, e)| Arc::strong_count(&e.value) == 1)
                .min_by_key(|(_, e)| e.last_used)
                .map(|(key, _)| *key);
            let Some(key) = victim else { break };
            let entry = state.entries.remove(&key).expect("victim was just found");
            state.resident -= entry.bytes;
            state.counters.evictions += 1;
        }
    }

    pub(crate) fn get_or_load<F>(
        self: &Arc<Self>,
        key: CacheKey,
        declared: usize,
        load: F,
    ) -> Result<CachePin<S>, KdfError>
    where
        F: FnOnce() -> Result<(Cached<S>, u64), KdfError>,
    {
        if declared > self.capacity || declared > self.in_flight_capacity {
            return Err(KdfError::ResourceLimit(format!(
                "decoded item requires {declared} bytes; cache={} in-flight={}",
                self.capacity, self.in_flight_capacity
            )));
        }
        let mut load = Some(load);
        loop {
            let mut state = self.state.lock().unwrap();
            if let Some(value) = state.entries.get(&key).map(|e| Arc::clone(&e.value)) {
                state.counters.hits += 1;
                Self::touch(&mut state, key);
                return Ok(CachePin {
                    value,
                    owner: Arc::downgrade(self),
                });
            }
            if state.loading.contains(&key) {
                state.counters.waits += 1;
                drop(self.changed.wait(state).unwrap());
                continue;
            }
            self.evict_unpinned(&mut state, declared);
            if state.resident + state.reserved + declared > self.capacity
                || state.in_flight + declared > self.in_flight_capacity
            {
                drop(self.changed.wait(state).unwrap());
                continue;
            }
            state.loading.insert(key);
            state.reserved += declared;
            state.in_flight += declared;
            state.counters.misses += 1;
            state.counters.peak_in_flight = state.counters.peak_in_flight.max(state.in_flight);
            drop(state);

            let result = load.take().expect("loader runs once")();
            let mut state = self.state.lock().unwrap();
            state.loading.remove(&key);
            state.reserved -= declared;
            state.in_flight -= declared;
            match result {
                Ok((value, compressed_bytes)) => {
                    let actual = value.bytes();
                    if actual != declared {
                        self.changed.notify_all();
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
                    state.entries.insert(
                        key,
                        Entry {
                            value: Arc::clone(&value),
                            bytes: actual,
                            last_used: 0,
                        },
                    );
                    Self::touch(&mut state, key);
                    self.changed.notify_all();
                    return Ok(CachePin {
                        value,
                        owner: Arc::downgrade(self),
                    });
                }
                Err(e) => {
                    self.changed.notify_all();
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
        let mut s = self.state.lock().unwrap();
        let (resident, in_flight) = (s.resident, s.in_flight);
        s.counters = Counters {
            peak_resident: resident,
            peak_in_flight: in_flight,
            ..Counters::default()
        };
    }

    pub(crate) fn stats(&self) -> KdfIoStats {
        let s = self.state.lock().unwrap();
        KdfIoStats {
            read_calls: s.counters.read_calls,
            compressed_bytes: s.counters.compressed_bytes,
            decoded_bytes: s.counters.decoded_bytes,
            cache_hits: s.counters.hits,
            cache_misses: s.counters.misses,
            evictions: s.counters.evictions,
            duplicate_load_waits: s.counters.waits,
            resident_bytes: s.resident,
            peak_resident_bytes: s.counters.peak_resident,
            in_flight_bytes: s.in_flight,
            peak_in_flight_bytes: s.counters.peak_in_flight,
            address_map_bytes: self.address_map_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One descriptor entry of `bytes` bytes, so capacities read as byte counts.
    fn load(bytes: usize) -> impl FnOnce() -> Result<(Cached<u8>, u64), KdfError> {
        move || Ok((Cached::Descriptor(vec![0u8; bytes]), bytes as u64))
    }

    fn get(cache: &Arc<Cache<u8>>, id: u32, bytes: usize) -> CachePin<u8> {
        cache
            .get_or_load(CacheKey::Descriptor(id), bytes, load(bytes))
            .expect("fits")
    }

    /// Eviction drops the least recently *used* entry, not the oldest loaded.
    ///
    /// This is the property the recency stamp exists for: re-reading an entry
    /// has to protect it from the next eviction, or a hot entry loaded early is
    /// thrown away while a cold one loaded later survives.
    #[test]
    fn eviction_picks_the_least_recently_used_entry() {
        // Room for two entries of ten bytes.
        let cache = Cache::<u8>::new(20, 1 << 20, 0);
        drop(get(&cache, 0, 10));
        drop(get(&cache, 1, 10));
        assert_eq!(cache.stats().read_calls, 2);

        // Touch 0, making 1 the least recently used.
        drop(get(&cache, 0, 10));
        assert_eq!(cache.stats().read_calls, 2, "0 should still be resident");

        // Admitting 2 must evict 1, the oldest *use*, and leave 0 alone.
        drop(get(&cache, 2, 10));
        assert_eq!(cache.stats().evictions, 1);
        drop(get(&cache, 0, 10));
        assert_eq!(cache.stats().read_calls, 3, "0 must not have been evicted");
        drop(get(&cache, 1, 10));
        assert_eq!(cache.stats().read_calls, 4, "1 must have been evicted");
    }

    /// A pinned entry is never the victim, even when it is the oldest.
    #[test]
    fn a_pinned_entry_survives_eviction() {
        let cache = Cache::<u8>::new(20, 1 << 20, 0);
        let held = get(&cache, 0, 10); // oldest, and kept pinned
        drop(get(&cache, 1, 10));

        drop(get(&cache, 2, 10));
        assert_eq!(cache.stats().evictions, 1);
        drop(held);
        // 0 was pinned so 1 went instead, despite being newer.
        drop(get(&cache, 0, 10));
        assert_eq!(cache.stats().read_calls, 3, "pinned 0 must have survived");
    }

    /// With every entry pinned there is no victim, and eviction gives up rather
    /// than spinning; the caller waits for a pin to be released instead.
    #[test]
    fn eviction_stops_when_everything_resident_is_pinned() {
        let cache = Cache::<u8>::new(20, 1 << 20, 0);
        let _a = get(&cache, 0, 10);
        let _b = get(&cache, 1, 10);
        let mut state = cache.state.lock().unwrap();
        cache.evict_unpinned(&mut state, 10);
        assert_eq!(state.counters.evictions, 0);
        assert_eq!(state.resident, 20);
    }
}
