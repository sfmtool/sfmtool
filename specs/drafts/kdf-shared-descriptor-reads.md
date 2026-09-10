# Reading a leaf's shared descriptors in one pass

**Status:** Draft. Amends
[core/features/lazy-kdforest-query.md](../core/features/lazy-kdforest-query.md)
§ "Two storage layouts, one search implementation".

The shared descriptor layout is 3.3x smaller than tree-local copies and faster in
every regime the layout benchmark measured but one: a fully resident warm batch,
where DinoLedge runs 0.04 s tree-local against 0.08 s shared. This draft proposes
closing that last cell.

Both figures are under a tenth of a second, so this is not urgent. It is worth
recording because the cause is known, the fix is contained, and it is the only
remaining argument for choosing tree-local.

## What is left

An earlier version of this draft also proposed making the cache's hit path O(1).
That shipped: `Cache::touch` no longer linear-searches a list of resident keys,
and per-hit cost is flat at ~0.19 us regardless of cache size. It was worth far
more than what remains here — the shared layout's warm DinoLedge batch went from
10.62 s to 0.08 s — and it is why the layout comparison now reads the way it does.

What it did not change is how *often* the shared path asks. For identical work on
`dino_dog_toy`, tree-local takes 275,022 cache hits and shared takes 541,727.
Halving that is the remaining gap.

## Where the extra lookups come from

Tree-local reads a leaf's descriptors as one contiguous slice of a chunk it is
already holding: one lookup per leaf, then a scan.
[`persistent.rs`](../../crates/sfmtool-core/src/features/kdforest/persistent.rs)
takes the shared path one feature at a time instead, calling `shared_vector` per
checked descriptor, which pins the containing block and then heap-allocates a
`Vec` to hand back a single 128-byte row. Every checked member of every leaf pays
a lookup and an allocation.

## The proposal

**Return a borrow, not a copy.** The pin already guarantees the block stays
resident for the duration of the access, so the descriptor can be read in place
and the per-descriptor allocation disappears. This is the simpler half and needs
no change to how the search visits a leaf.

**Group a leaf's reads by block.** A leaf's feature IDs map to storage rows that
are scattered, but not uniformly: rows are assigned in tree-0 leaf order, so
tree-0 leaves are contiguous and other trees' leaves cluster to the extent their
partitions agree. Sorting a leaf's IDs by block and holding one pin per distinct
block turns a per-descriptor lookup into a per-block one. Nothing on disk moves;
only the visit order within one leaf changes, and the result set is
order-independent.

The check count, the neighbors and the distances must all be unchanged. This is
an access-path change, and the parity tests that hold the file-backed search to
the in-memory one already assert exactly that.

## What would settle it

Re-run the resident case from the benchmark method. If the shared layout's warm
batch reaches tree-local's, the last reason to prefer tree-local goes with it and
the choice is purely about whether a caller wants the smaller file. If a gap
survives, it is worth knowing how much of it is the descriptor scatter itself,
which no amount of batching removes.
