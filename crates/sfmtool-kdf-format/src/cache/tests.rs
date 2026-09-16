// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// One descriptor entry of `bytes` bytes, so capacities read as byte counts.
fn load(bytes: usize) -> impl FnOnce() -> Result<(Cached<u8>, u64), KdfError> {
    move || Ok((Cached::Descriptor(vec![0u8; bytes]), bytes as u64))
}

fn get(cache: &Arc<Cache<u8>>, id: u32, bytes: usize) -> CachePin<'_, u8> {
    cache
        .get_or_load(CacheKey::Descriptor(id), bytes, load(bytes))
        .expect("fits")
}

/// Isolate cache CPU cost from file I/O and decompression.
#[test]
#[ignore = "manual release-mode performance measurement"]
fn benchmark_cache_churn() {
    for count in [1024u32, 16384, 65536] {
        let cache = Cache::<u8>::new(count as usize, count as usize, 0, count as usize);
        for id in 0..count {
            drop(get(&cache, id, 1));
        }
        let start = std::time::Instant::now();
        for id in count..count + 10000 {
            drop(get(&cache, id, 1));
        }
        eprintln!(
            "churn resident={count} ns/miss={:.0}",
            start.elapsed().as_nanos() as f64 / 10000.0
        );
        let start = std::time::Instant::now();
        for _ in 0..100 {
            for id in count + 9000..count + 10000 {
                drop(get(&cache, id, 1));
            }
        }
        eprintln!(
            "hit resident={count} ns/hit={:.0}",
            start.elapsed().as_nanos() as f64 / 100000.0
        );
    }
}

/// Eviction drops the least recently *used* entry, not the oldest loaded.
///
/// This is the property the recency list exists for: re-reading an entry
/// has to protect it from the next eviction, or a hot entry loaded early is
/// thrown away while a cold one loaded later survives.
#[test]
fn eviction_picks_the_least_recently_used_entry() {
    // Room for two entries of ten bytes.
    let cache = Cache::<u8>::new(20, 1 << 20, 0, 20);
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
    let cache = Cache::<u8>::new(20, 1 << 20, 0, 20);
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
    let cache = Cache::<u8>::new(20, 1 << 20, 0, 20);
    let _a = get(&cache, 0, 10);
    let _b = get(&cache, 1, 10);
    let shard = &cache.shards[0];
    let mut state = shard.state.lock().unwrap();
    Cache::evict_unpinned(shard, &mut state, 10);
    assert_eq!(state.counters.evictions, 0);
    assert_eq!(state.resident, 20);
}

#[test]
fn releasing_the_last_pin_wakes_admission() {
    let cache = Cache::<u8>::new(10, 10, 0, 10);
    let held = get(&cache, 0, 10);
    let other = Arc::clone(&cache);
    let (send, recv) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        drop(get(&other, 1, 10));
        send.send(()).unwrap();
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while cache.shards[0].waiters.load(Ordering::SeqCst) == 0 {
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    drop(held);
    recv.recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    worker.join().unwrap();
}

#[test]
fn shards_respect_the_total_in_flight_limit() {
    let cache = Cache::<u8>::new(1 << 20, 1024, 0, 1024);
    assert_eq!(cache.shards.len(), MAX_SHARDS);
    let active = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for id in 0..32 {
            let cache = &cache;
            let active = &active;
            scope.spawn(move || {
                drop(
                    cache
                        .get_or_load(CacheKey::Descriptor(id), 1024, || {
                            assert_eq!(active.fetch_add(1, Ordering::SeqCst), 0);
                            std::thread::sleep(std::time::Duration::from_millis(1));
                            active.fetch_sub(1, Ordering::SeqCst);
                            Ok((Cached::Descriptor(vec![0u8; 1024]), 1024))
                        })
                        .unwrap(),
                );
            });
        }
    });
    assert_eq!(cache.stats().peak_in_flight_bytes, 1024);
    assert_eq!(cache.stats().in_flight_bytes, 0);
}

/// Shards are sized so each can hold the largest item a caller may ask for.
///
/// Admission is per shard, so a shard smaller than one item could never admit
/// it and the caller would block forever. The count is therefore capped by
/// capacity / largest_item, not chosen freely.
#[test]
fn shard_count_never_starves_the_largest_item() {
    // Room for 64 items of 1 KiB: capped by MAX_SHARDS, not by capacity.
    let plenty = Cache::<u8>::new(64 << 10, 1 << 20, 0, 1 << 10);
    assert_eq!(plenty.shards.len(), MAX_SHARDS);
    assert!(plenty.shards[0].capacity >= 1 << 10);

    // Room for only two such items: two shards, each still able to hold one.
    let tight = Cache::<u8>::new(2 << 10, 1 << 20, 0, 1 << 10);
    assert_eq!(tight.shards.len(), 2);
    assert!(tight.shards[0].capacity >= 1 << 10);

    // Room for exactly one: a single shard, which is the unsharded cache.
    let single = Cache::<u8>::new(1 << 10, 1 << 20, 0, 1 << 10);
    assert_eq!(single.shards.len(), 1);
    assert_eq!(single.shards[0].capacity, 1 << 10);
}

#[test]
fn tree_roots_spread_across_shards() {
    let cache = Cache::<u8>::new(64 << 10, 1 << 20, 0, 1 << 10);
    let roots: std::collections::HashSet<_> = (0..4)
        .map(|tree| cache.shard_index_of(CacheKey::Tree(tree, 0)))
        .collect();
    assert_eq!(roots.len(), 4);
}

/// Consecutive block indexes land on different shards.
///
/// Descriptor blocks are read in stored order, so a run of neighbouring
/// blocks is exactly what concurrent workers contend over; spreading those
/// across shards is the whole point. Keys are dense integers, so the low bits
/// do it without hashing.
#[test]
fn neighbouring_blocks_spread_across_shards() {
    let cache = Cache::<u8>::new(MAX_SHARDS << 10, 1 << 20, 0, 1 << 10);
    assert_eq!(cache.shards.len(), MAX_SHARDS);
    let used: std::collections::HashSet<usize> = (0..MAX_SHARDS as u32)
        .map(|b| cache.shard_index_of(CacheKey::Descriptor(b)))
        .collect();
    assert_eq!(used.len(), MAX_SHARDS, "each block took its own shard");
}

/// An item larger than a shard is refused, rather than waited on forever.
#[test]
fn an_item_larger_than_its_shard_is_refused() {
    let cache = Cache::<u8>::new(64 << 10, 1 << 20, 0, 1 << 10);
    let big = cache.shards[0].capacity + 1;
    let refused = cache.get_or_load(CacheKey::Descriptor(0), big, || {
        Ok((Cached::Descriptor(vec![0u8; big]), big as u64))
    });
    match refused {
        Err(KdfError::ResourceLimit(_)) => {}
        Err(other) => panic!("wrong error: {other:?}"),
        Ok(_) => panic!("an item larger than its shard must be refused"),
    }
}
