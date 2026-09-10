# Batching shared descriptor reads

**Status:** Draft. Amends
[core/features/lazy-kdforest-query.md](../core/features/lazy-kdforest-query.md)
§ "Two storage layouts, one search implementation".

The shared descriptor layout stores 3.3x less than tree-local copies and moves
less data per query, but is markedly slower whenever the file is resident in the
reader's cache. The benchmark that found this also found that the gap is not
about storage at all: it is the per-descriptor cost of the shared read path.
This draft proposes closing it.

## What the measurement showed

On DinoLedge — 9.7M descriptors, four trees — with a 4 GiB cache holding the
whole file, a warm 1,000-query batch takes 0.54 s tree-local and 10.62 s shared.
Both batches issue **zero reads**: every access is a cache hit, so no I/O
separates them. What differs is how many accesses there are, and what each one
costs.

| | tree-local | shared |
|---|---|---|
| Cache lookups, warm batch | 185,501 | 317,372 |
| Reads, warm batch | 0 | 0 |
| Decoded bytes, cold batch | 3,245.7 MiB | 1,457.6 MiB |

The shared layout touches the cache 1.7x more often while decoding 2.2x *less*
data. A 20x wall-clock gap does not follow from a 1.7x lookup count, so the cost
is per lookup.

## Where it goes

Tree-local reads a leaf's descriptors as one contiguous slice of a chunk it is
already holding: one cache lookup per leaf, then a scan.
[`persistent.rs`](../../crates/sfmtool-core/src/features/kdforest/persistent.rs)
takes the shared path one feature at a time instead, and each iteration calls
`shared_vector`, which

1. takes the cache mutex to pin the descriptor's block, and
2. heap-allocates a `Vec` to return one 128-byte descriptor.

Both happen once per *checked descriptor*, and the leaf loop checks every
member. At 317,372 checks that is 317,372 mutex acquisitions and 317,372
allocations of 128 bytes each, to read data already in memory.

## The proposal

Neither cost is inherent to storing descriptors once.

**Return a borrow, not a copy.** The pin already guarantees the block stays
resident for the duration of the access, so the descriptor can be read in place.
This removes the allocation without changing what is stored, and is the larger
and simpler half.

**Group a leaf's reads by block.** A leaf's feature IDs map to storage rows that
are scattered, but not uniformly: they were assigned in tree-0 leaf order, so
leaves of tree 0 are contiguous and other trees' leaves cluster to the extent
their partitions agree. Sorting a leaf's IDs by block and holding one pin per
distinct block turns a per-descriptor lookup into a per-block one. The
descriptors do not need reordering on disk for this; only the visit order within
one leaf changes, and the result set is order-independent.

The check count, the neighbors and the distances must all be unchanged — this is
an access-path change, and the parity tests that hold the file-backed search to
the in-memory one already assert exactly that.

## What would settle it

Re-run the resident case from the benchmark method with both changes in place.
The question is whether shared's warm batch comes within a small factor of
tree-local's; if it does, the layout choice returns to being about size and cold
behavior, where shared already wins. If a large gap survives, the shared layout
carries a real query-time cost and the default should say so.

Until then no shipping default is chosen, because the number a default would be
chosen on is the one this draft expects to move.
