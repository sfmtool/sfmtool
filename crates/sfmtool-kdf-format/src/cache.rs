// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet, VecDeque};
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
    lru: VecDeque<CacheKey>,
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
                lru: VecDeque::new(),
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

    fn touch(state: &mut State<S>, key: CacheKey) {
        if let Some(i) = state.lru.iter().position(|&k| k == key) {
            state.lru.remove(i);
        }
        state.lru.push_back(key);
    }

    fn evict_unpinned(&self, state: &mut State<S>, needed: usize) {
        let mut examined = 0;
        while state
            .resident
            .saturating_add(state.reserved)
            .saturating_add(needed)
            > self.capacity
            && examined < state.lru.len()
        {
            let key = state.lru.pop_front().expect("length checked");
            let pinned = state
                .entries
                .get(&key)
                .is_some_and(|e| Arc::strong_count(&e.value) > 1);
            if pinned {
                state.lru.push_back(key);
                examined += 1;
            } else if let Some(entry) = state.entries.remove(&key) {
                state.resident -= entry.bytes;
                state.counters.evictions += 1;
                examined = 0;
            }
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
