# The cache's per-hit cost, and the shared layout's hit count

**Status:** Draft. Amends
[core/features/lazy-kdforest-query.md](../core/features/lazy-kdforest-query.md)
§ "Reads, cache, and memory accounting".

The shared descriptor layout stores 3.3x less than tree-local copies and moves
less data per query, but is markedly slower whenever the file is already resident
in the reader's cache. The layout benchmark traced that gap to two things, and
neither is about storage: **the cache costs O(resident entries) per hit**, and the
shared read path incurs about twice as many hits as it needs. The first is a
performance bug affecting both layouts; the second is what makes the first bite
harder in the shared case.

## What the measurement showed

On DinoLedge — 9.7M descriptors, four trees — with a 4 GiB budget holding the
whole file, a warm 1,000-query batch takes 0.54 s tree-local and 10.62 s shared.
Both batches issue **zero reads**: every access is a cache hit, so no I/O
separates them. That works out to 2.9 us per hit tree-local and 33.5 us shared,
against the tens of nanoseconds a mutex and a hash lookup should cost. Something
in the hit path is doing real work.

Holding hits and reads fixed and varying only how many objects the cache holds
isolates it. Same corpus (`dino_dog_toy`, 694k descriptors), same forest, same
queries, a budget large enough that every pass is pure cache hits, varying only
the shared descriptor block size:

| Layout | ZIP entries | Cache hits | Reads | Warm batch | Per hit |
|--------|------------|-----------|-------|-----------|---------|
| tree-local | 2,066 | 275,022 | 0 | 0.10 s | 0.38 us |
| shared, 1 MiB blocks | 1,636 | 541,727 | 0 | 0.24 s | 0.44 us |
| shared, 256 KiB blocks | 1,891 | 541,727 | 0 | 0.38 s | 0.69 us |
| shared, 64 KiB blocks | 2,908 | 541,727 | 0 | 0.59 s | 1.09 us |
| shared, 16 KiB blocks | 6,976 | 541,727 | 0 | 1.93 s | 3.57 us |

The hit count is identical across the four shared rows and no row reads a byte.
The only thing changing is how many objects the cache is holding, and the cost of
a hit tracks it — 4.3x more entries, 8.1x the per-hit cost.

## Where it goes

`Cache::touch` runs on every hit, inside the cache mutex:

```rust
fn touch(state: &mut State<S>, key: CacheKey) {
    if let Some(i) = state.lru.iter().position(|&k| k == key) {
        state.lru.remove(i);
    }
    state.lru.push_back(key);
}
```

The recency list is a `VecDeque<CacheKey>` carrying one element per resident
object, so promoting an entry is a linear search followed by a `VecDeque::remove`
that shifts the tail. Every hit pays it. On the DinoLedge shared file the list
holds roughly 23,000 blocks, which is how a hash-map lookup turns into 33.5 us.

That cost is the same in kind for both layouts — tree-local is only cheaper
because it holds a few thousand large chunks rather than tens of thousands of
small blocks, and asks half as often.

The second half is the asking. Tree-local reads a leaf's descriptors as one
contiguous slice of a chunk it is already holding: one lookup per leaf, then a
scan.
[`persistent.rs`](../../crates/sfmtool-core/src/features/kdforest/persistent.rs)
takes the shared path one feature at a time instead, calling `shared_vector` per
checked descriptor, which pins the containing block and then heap-allocates a
`Vec` to hand back a single 128-byte row. Hence 541,727 hits against tree-local's
275,022 for identical work.

## The proposal

**Make the hit path O(1).** Give `Entry` a `last_used: u64`, give `State` a
counter to stamp it from, and delete the `VecDeque`. `touch` becomes a field
write. Eviction then picks its victim by scanning `entries` for the smallest
stamp among unpinned ones, which is O(resident) — but eviction already walks the
recency list re-queueing pinned entries, so this moves cost from the common path
to the rare one rather than adding it. Even in the most thrashing configuration
measured, hits outnumbered evictions three to one; in the resident case there are
no evictions at all. This is contained to
[`cache.rs`](../../crates/sfmtool-kdf-format/src/cache.rs), changes no public
API, and touches no wire format.

**Then halve the shared layout's hit count.** Two changes, both local to the
shared read path:

*Return a borrow, not a copy.* The pin already guarantees the block stays
resident for the duration of the access, so the descriptor can be read in place
and the per-descriptor allocation disappears.

*Group a leaf's reads by block.* A leaf's feature IDs map to storage rows that
are scattered, but not uniformly: rows were assigned in tree-0 leaf order, so
tree-0 leaves are contiguous and other trees' leaves cluster to the extent their
partitions agree. Sorting a leaf's IDs by block and holding one pin per distinct
block turns a per-descriptor lookup into a per-block one. Nothing on disk moves;
only the visit order within one leaf changes, and the result set is
order-independent.

The check count, the neighbors and the distances must all be unchanged — this is
an access-path change, and the parity tests that hold the file-backed search to
the in-memory one already assert exactly that.

## What would settle it

Re-run the resident case from the benchmark method. The first change should move
both layouts and the second should close most of what remains between them. If
shared then lands within a small factor of tree-local, the layout choice returns
to being about size and cold behavior, where shared already wins on both. If a
large gap survives, the shared layout carries a real query-time cost and a
default should say so.

Until then no shipping default is chosen, because the number a default would be
chosen on is the one this draft expects to move.
