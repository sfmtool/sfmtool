# Lazy KD-Forest Queries

A persistent kd-tree forest lets a process search a large set of descriptors
while keeping only the portions it visits in memory. The file-backed query path
in core does this over chunked trees with either leaf-local vectors or a shared
vector table, targeting local seekable files, repeated queries, and corpora
larger than the configured memory cache. It preserves the in-memory forest's
search behavior in both layouts.

Packing and cache values are configurable. The starting values below are chosen
for plausibility rather than measurement; the benchmark plan that would settle
them is in [Benchmark plan and provisional defaults](#benchmark-plan-and-provisional-defaults),
and no benchmark results are claimed here.

It extends [randomized kd-tree forests](randomized-kdtree-forest.md)
and uses the [KDF format](../../formats/kdf-file-format.md).

## Rust interface and responsibilities

The integration belongs beside the existing implementation in
[core's kdforest module](../../../crates/sfmtool-core/src/features/kdforest/mod.rs).
The `sfmtool-kdf-format` crate owns storage types and validated decoded chunks;
core depends on it, never the reverse. Persistence exports the already-built
topology and feature order rather than rebuilding from a seed. This avoids making
random-generator or future builder changes part of the file compatibility contract.

The public surface is implemented in
[`kdforest/persistent.rs`](../../../crates/sfmtool-core/src/features/kdforest/persistent.rs)
and re-exported by the kdforest module:

```rust
pub struct KdfWriteOptions {
    pub descriptor_storage: DescriptorStorage, // explicit choice; no default yet
    pub target_chunk_bytes: usize, // provisional: 1 MiB
    pub compression_level: i32,    // provisional: 3
    pub origin_block_rows: usize,  // provisional: 131072 (two u32 columns = 1 MiB)
}
pub struct LazyKdForestOptions {
    pub max_address_map_bytes: usize, // provisional: 256 MiB, shared layout only
    pub max_leaf_features: usize,     // provisional: 1,048,576
    pub cache_bytes: usize,        // provisional: 256 MiB decoded cache
    pub max_in_flight_bytes: usize,// provisional: 64 MiB decode reservations
    pub max_compressed_bytes: usize, // provisional: 64 MiB per-entry compressed scratch
    pub max_metadata_bytes: usize, // provisional: 64 MiB
    pub max_chunk_bytes: usize,    // provisional: 64 MiB decoded
    pub query_workers: usize,     // provisional: 1; caller can raise
}
pub struct LazyKdForest<S: ForestScalar + KdfScalar> { /* file handle, cache, workers */ }
pub enum DescriptorStorage {
    TreeLocal,
    Shared { target_descriptor_block_bytes: usize },
}
pub type LazyKdForestU8 = LazyKdForest<u8>;
pub type LazyKdForestF32 = LazyKdForest<f32>;

impl<S: KdfScalar> KdForest<S> {
    pub fn write_kdf(&self, path: &Path, sources: Option<&KdfSiftSources>,
                     options: &KdfWriteOptions)
        -> Result<(), KdfError>;
}
impl<S: KdfScalar> LazyKdForest<S> {
    pub fn open(path: &Path, options: LazyKdForestOptions)
        -> Result<Self, KdfError>;
    pub fn search(&self, query: &[S], k: usize, max_leaf_checks: usize,
                  max_dist: Option<f32>) -> Result<Vec<Neighbor>, KdfError>;
    pub fn search_with_stats(&self, query: &[S], k: usize, max_leaf_checks: usize,
        max_dist: Option<f32>) -> Result<(Vec<Neighbor>, LazyQueryStats), KdfError>;
    pub fn search_batch_with_distances(&self, queries: &[S], n_queries: usize,
        k: usize, max_leaf_checks: usize, max_dist: Option<f32>)
        -> Result<(Vec<u32>, Vec<f32>), KdfError>;
    pub fn len(&self) -> usize;
    pub fn dim(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn resolve_origins(&self, feature_ids: &[u32])
        -> Result<Option<Vec<FeatureOrigin>>, KdfError>;
    pub fn image_table(&self) -> Result<Option<&KdfImageTable>, KdfError>;
    pub fn io_stats(&self) -> KdfIoStats;
}
```

`search_with_stats` returns `LazyQueryStats` — leaf checks, heap pushes and pops
— alongside the neighbors; `search` is the same traversal with the counters
dropped. The counters exist because they are what the parity tests assert on:
matching neighbor IDs alone would not catch a file-backed traversal that visits
a different set of leaves and happens to agree. `io_stats` reports the cache and
read counters of `KdfIoStats` for the whole file, which is how a test asserts
that opening reads no chunk payload and that a warm hit causes no read.

`KdfScalar` is a sealed bridge for the existing u8/f32 scalar implementations,
not an invitation to persist arbitrary user metrics. `Neighbor` retains original
row ID and squared distance. Fallible queries are necessary because an unread
chunk can fail after open succeeds. Errors distinguish I/O (with entry context),
unsupported format/scalar, invalid shape/query/options, resource limit, malformed
structure and integrity mismatch. A failed query or batch returns an error, not
an apparently successful shortened result.

Illustrative usage, with a file built from the same in-memory index:

```rust
use sfmtool_core::features::kdforest::{
    KdForestU8, KdForestParams, KdfWriteOptions,
    LazyKdForestU8, LazyKdForestOptions,
};
let features = vec![0u8, 0, 10, 10, 1, 1];
let forest = KdForestU8::build(&features, 3, 2, KdForestParams::balanced());
let options = KdfWriteOptions::tree_local();
forest.write_kdf("example.kdf".as_ref(), None, &options)?;
let lazy = LazyKdForestU8::open("example.kdf".as_ref(), Default::default())?;
let neighbors = lazy.search(&[0, 0], 2, 128, None)?;
assert_eq!(neighbors[0].index, 0);
```

Storage-level indexed read/write/full-verify interfaces belong to `sfmtool-kdf-format`;
they accept neutral trees/chunks and expose no ANN algorithm. Those APIs stay
storage-specific rather than forming a general plugin surface.
Writing uses a sibling temporary file and publishes only a completed archive;
it fails if the destination exists. Streaming construction from an out-of-memory
input corpus is out of scope: export requires an already-built forest.

### Source references

`KdfSiftSources` contains the format's workspace configuration, image names and
both per-image hashes, plus parallel image/image-feature index arrays of length N in
original input order. Writers validate complete coverage and pair uniqueness.
`FeatureOrigin` contains `image_index: u32` and `image_feature_index: u32`; an image
table accessor exposes the names and hashes without opening source files.
`resolve_origins` returns mappings in requested ID order, including repeated
requests, returns `None` for a generic corpus, and rejects out-of-range IDs.

Origin blocks are cached on demand under the same byte budget as tree chunks;
only blocks covering requested result IDs are read. Image metadata/hashes load
on first source-table access, within the metadata budget. Normal ANN reads
neither origins nor image tables. Offline KDF verification checks origin ranges
and uniqueness; a separate explicit source verifier checks referenced SIFT hashes,
feature bounds and vector equality, reporting missing files distinctly.
Repacking preserves this mapping. Exporting a descriptor subset retains its
original SIFT feature indices even though input row IDs are newly dense.

## Two storage layouts, one search implementation

Implement both layouts in the first delivery. `KdfWriteOptions::tree_local()`
and `KdfWriteOptions::shared(target_descriptor_block_bytes)` set the existing
provisional compression, origin and tree-chunk defaults. Require callers to
choose a layout until benchmarks establish a default. Open detects the layout
from metadata; query signatures and results are identical for both.

Keep one best-bin-first traversal, result set, dedup set and scalar distance
implementation. Storage access supplies node data, ordered leaf feature IDs and
vector rows. In tree-local mode, a vector row is in the leaf's chunk; in shared
mode, its feature ID indexes the resident storage-row map, which locates a shared
descriptor block. This adds a storage access strategy, not a second ANN algorithm.
Use the same original forest for both exports rather than rebuilding it twice.

Shared export orders descriptors by tree 0's leaf permutation, writes its inverse
as storage_rows, and partitions vectors into Q complete rows per block, where
`Q = max(1, floor(target_descriptor_block_bytes / bytes_per_vector))`. Reject zero
targets. Targets smaller than a row produce one-row blocks. This block target is
independent of the tree-chunk target and does not change topology or leaf size.

Open loads and validates the shared row map within `max_address_map_bytes`,
checking permutation validity with bounded temporary memory. Reject excess
rather than silently paging it. Report resident map and validation-scratch bytes;
the map is separate from the decoded chunk cache and metadata limit. Tree-local
mode allocates no map.

Shared descriptor blocks, tree chunks and origin blocks use the same byte-weighted
cache with distinct key kinds, integrity checks and single-load coordination.
Before requesting shared descriptors, copy the ordered leaf IDs into query scratch
and release the tree chunk pin. Process rows in leaf order, deduplicating IDs before
fetching vectors; release a descriptor block pin before requesting another block.
This prevents waiting for cache admission while retaining a different cache pin.
Leaf scratch is additional memory bounded by `max_leaf_features`; reject oversized
leaves. Prefetching or grouping reads must not reorder evaluations or result ties.

The extra implementation work is the shared writer/address map, vector-block
reader, cache key variant, layout-specific validation/hashing, and cross-layout
tests. No second ANN algorithm or source mapping is needed. Keep the storage
interface private initially rather than designing general-purpose plugins.

## Packing policy

The format permits arbitrary chunk boundaries. The baseline exporter computes
subtree packing weights including nodes, split values, IDs and vector copies.
Use these same weights in both layouts so identical options produce identical
tree partitions for a controlled comparison. In shared mode vector bytes are
virtual packing weight only: stored decoded_bytes records the smaller actual
tree arrays. Shared vector blocks use their separate target. A future policy
may pack shared trees more tightly without changing format semantics.
Starting at the root, every maximal complete subtree that fits the target
becomes a chunk. An oversized leaf forms a chunk of its own. The internal nodes
above those subtrees form routing chunks, packed in deterministic breadth-first
order up to the same byte target. Child references connect the routing chunks
and complete-subtree chunks. The root is therefore reachable through small,
reusable routing data; fetching a complete-subtree chunk finishes any descent
inside it without another descriptor fetch.

Local node order is deterministic (preorder within a complete subtree). Leaf
rows follow local leaf order, preserving each leaf's original member order.
Logical node IDs retain source arena IDs even when physical node order changes.
Assign chunk IDs deterministically, root chunk first, then remaining chunks in
left-before-right traversal order. Physical ZIP order follows tree/chunk IDs.
No writer changes leaf size to make chunks fit: that changes the ANN index.

For a binary median tree, maximal fitting subtrees often underfill the target;
report actual size distributions instead of labeling every chunk “1 MiB”.
Never pack by compressed size: compression depends on data and would make
allocation sizes and benchmark comparisons misleading.

The per-tree vector copies are intentional. A shared vector table in original
ID order can scatter one 16-feature leaf over 16 descriptor chunks. Reordering a
shared table by tree 0 improves that tree, but gives no corresponding guarantee
for the other randomized trees — measured, that is 89% block reuse at one tree
falling to 31% at four, in
[What the measurements found](#what-the-measurements-found). Duplication trades disk
capacity and write time for predictable leaf access. It is the main review decision,
not a free gain.

## Search behavior and parity

Use the current in-memory search as the behavioral reference, specifically
[search.rs](../../../crates/sfmtool-core/src/features/kdforest/search.rs), rather
than reproducing the older pseudocode's equality convention. The storage path
must preserve the following details:

- Seed every tree in increasing tree index before testing the check budget.
  Descend left for query coordinate <= split, right for > split. For float32,
  use the existing scalar total-order comparison, including signed-zero behavior.
- A descent retains its incoming branch priority along its near chain; enqueue
  far children with incoming priority plus squared split-plane distance, only
  when this is <= the current result threshold.
- Pop the smallest priority first. Equal priorities use descending tree index,
  then descending **logical node ID**, matching the existing heap tuple order.
  Chunk IDs must never replace logical node IDs in this comparison.
- Deduplicate original feature IDs across trees and evaluate leaf members in their
  stored order. Check count measures unique distance evaluations. Finish a leaf
  once entered, even when it crosses the budget; zero checks still seeds trees.
- After seeding, stop before the next descent if checks meet the budget or its
  priority exceeds the current threshold. The additive priority is not an
  admissible geometric bound: unlimited checks still do not imply exact ANN.
- Preserve result tie behavior: equal distances retain encounter order; an equal
  candidate does not replace the worst member of a full result set. Use the
  same scalar kernels and cutoff conversion as the in-memory implementation.

Cache hits, eviction, file layout, and worker schedule affect speed only.
For the same persisted topology and supported finite inputs, results and check
counts match the in-memory reference. Float results are bit-identical when
using the same scalar kernel/platform; no cross-platform float promise is added.
Reject NaN/infinite query coordinates and negative/NaN cutoffs; positive infinity
means unbounded. Finite float coordinates may overflow squared distance to
positive infinity, as in the existing kernel. Empty forests and k = 0 return
empty results without reading tree chunks. A wrong query dimension is an error.
Batch arrays are row-major with `u32::MAX`/positive infinity padding.

There is no independent I/O cutoff that silently truncates ANN. A future
latency-budgeted search would need to report incompleteness explicitly.

## Reads, cache, and memory accounting

Open reads the ZIP central directory, metadata and content-hash entry once,
builds a chunk-to-entry index, and validates their bounded sizes. It does not
decode trees or descriptors. ZIP metadata/index memory is O(number of entries),
not constant; enforce the metadata budget on both decoded JSON and index
allocations, rejecting excess directory entries before unbounded allocation.

On a miss, read only that chunk's three tree-entry ranges, plus its vector entry
in tree-local layout, and verify/decode them.
When entries are adjacent, a reader may coalesce their ranges, including intervening
ZIP headers. A chunk is a logical cache unit, not necessarily one system call.
Cache the offset/length index for the handle's lifetime; never reopen or reparse
the ZIP for a node. Positional reads or separately positioned handles avoid a
shared seek cursor race. A lock around seek/read is an acceptable first fallback;
decompression happens outside it. Do not use the existing eager `DecodedEntries`
path to load the whole archive.

Use a shared byte-weighted LRU of decoded chunks, keyed by file handle identity,
tree and chunk. In-flight requests for the same chunk share one load. Keep no
unbounded pinned “upper tree”: frequent routing chunks stay hot through reuse.
Each descent holds at most its current decoded chunk and releases it before
requesting another. Queue entries hold addresses, not chunk references.

The cache budget includes pinned resident chunk arrays; admission reserves space
and waits for readers to release chunks if necessary. Require cache and in-flight
limits each to admit the largest declared chunk, and reject declared chunks over
`max_chunk_bytes`. Apply the same limits to origin blocks. Reserve decode memory
before I/O. Separately bound compressed
buffers and decoder workspace; include these in reported peak memory rather
than claiming the decoded cache limit is a process RSS limit. A loader never
waits for admission while holding a different chunk pin, avoiding cache deadlock.

Per-query dedup uses a sparse set of visited original IDs, so scratch grows with
work performed rather than allocating the in-memory implementation's N-bit array
per worker. Queue memory grows with explored branches. Batch output is O(M*k);
metadata, scratch, output, compressed buffers and decoder workspace are additional
to the cache. Checked arithmetic and configurable worker count constrain growth;
this is bounded residency of file data, not constant total memory for arbitrary k.

Treat open files as immutable for the handle lifetime. Atomic replacement can
leave an old handle serving its old snapshot; in-place writes are unsupported.
Chunk validation checks references before dereference and detects revisited
logical nodes per query to prevent malformed cycles. Full semantic verification
is an explicit offline operation, never implicit at lazy open.

## Benchmark method

One MiB is 1,048,576 decoded bytes. Start with a configurable 1 MiB target,
then compare 256 KiB, 1, 4, 8 and 16 MiB; these are experiment settings, not
claims that one size is optimal. Larger chunks reduce directory entries and may
amortize reads across queries, but increase cold-query read/decode amplification.
In particular, four independent 16 MiB subtree misses can decode 64 MiB merely
to seed a four-tree search. Directory lookup itself is paid once at open.

Hold vectors, topology, query order, k and check budget fixed when comparing
packing. Compare both supported version-1 layouts, with the shared corpus packed
in tree-0 order, and eager loading of the same forest. Sweep shared descriptor
block sizes independently at 16, 64, 256 KiB and 1 MiB.

| Axis | Cases |
|------|-------|
| Corpus | Real 128-D SIFT at 100k, 1M, and a size exceeding RAM; clustered/duplicate byte data; finite 32-D and 128-D float data |
| Forest | 4, 8 and 20 trees; leaf sizes 8, 16, 32; checks 32, 128, 512; k = 1, 2, 10 |
| Query | Independent held-out queries; self-query with self excluded for recall; random and locality-grouped batches; single-query and large batches |
| Cache | 64, 256, 1024 MiB where limits admit chunks; cache smaller than total corpus; worker counts 1, 4, 8 |
| Storage | Local SSD; HDD if available; record OS, device, filesystem and compression level |
| Warmth | Fresh process/application cache with warm OS cache; controlled cold OS cache where available; fully warm repeat |

Use a staged sweep, not the entire Cartesian product: screen chunk sizes on
four trees/16-feature leaves first, then stress the winning candidates. A reopened
file is not evidence of cold physical storage; label uncontrolled cache state.
Do not download benchmark datasets automatically as part of implementing specs.

Measure open time and resident metadata, p50/p95/p99 query latency, batch
throughput, compressed bytes requested, decoded bytes, read calls, unique chunk
misses, cache hits/evictions, duplicate-load suppression, peak cache/pinned/
in-flight/scratch/RSS bytes, file size and export time. Application read bytes
are distinct from physical device traffic. Report read amplification as decoded
chunk bytes divided by unique evaluated vector bytes (undefined for zero checks).

Measure recall against exhaustive search on a held-out subset, and separately
assert exact result/check parity with the in-memory forest. Excluding self for
recall must use the same postprocessing/reference procedure for both paths;
it does not add an exclusion option to the API. Repeat runs, report variability,
and record corpus/query hashes and hardware.

[`scripts/benchmark_kdf_layouts.py`](../../../scripts/benchmark_kdf_layouts.py)
runs this against a workspace's `.sift` files. It builds the forest once per run
and exports it repeatedly, so every cell in a sweep compares storage against an
identical forest, query set, k and check budget.

Read amplification is measured on a **single** cold query, not on a batch.
Within one query the check set is deduplicated, so `checks x dim` is exactly the
unique evaluated vector bytes the ratio divides into; across a batch it is not,
because later queries re-evaluate descriptors already decoded. Batch numbers
below therefore report decoded bytes and read counts directly.

## What the measurements found

Measured on the three corpora below at four trees and 16-feature leaves, k = 2, a
128-leaf check budget and zstd level 3. Timings are medians of repeated runs on
one Windows desktop with a local NVMe SSD; the OS page cache is uncontrolled, so
"cold" means a fresh reader with an empty application cache and nothing stronger.

| Corpus | Images | Descriptors | Tree-local | Shared | Ratio |
|--------|--------|-------------|-----------|--------|-------|
| `seoul_bull_sculpture` | 17 | 35,167 | 13.94 MiB | 4.08 MiB | 3.42x |
| `dino_dog_toy` | 85 | 694,320 | 262.18 MiB | 78.07 MiB | 3.36x |
| DinoLedge | 1,196 | 9,701,948 | 3,840.61 MiB | 1,157.26 MiB | 3.32x |

**One shared corpus is consistently 3.3-3.4x smaller** than four tree-local
copies, across corpora spanning 276x in size. It falls short of the naive 4x
because the tree node, split and feature-ID arrays are stored either way and the
shared layout adds a feature-ID-to-row map — 32.8 MB compressed at 9.7M features,
against 2.8 GB saved.

**Descriptors compress to 76.4% in kd-tree leaf order.** The projections in
[kdf-file-format.md](../../formats/kdf-file-format.md) had to estimate this from
image order and a random shuffle, and said neither bounded leaf order. Leaf order
sits just below image order's 76.96-76.98%, so the proxy was sound.

**The first thing the benchmark found was a cache defect, not a layout result.**
`Cache::touch` promoted an entry on every hit by linear-searching a list of every
resident key, making a cache hit O(resident entries) — 33.5 us per fully-cached
access on a file holding ~23,000 descriptor blocks. That charged the shared
layout hardest, because it holds many small blocks rather than a few large chunks
and resolves each descriptor separately. Fixed in
[`cache.rs`](../../../crates/sfmtool-kdf-format/src/cache.rs) by stamping entries
with a counter; per-hit cost is now flat at ~0.19 us regardless of cache size.
Every number below is post-fix, and the layout comparison inverted when it
landed: read the pre-fix figures in this file's history as a measurement of that
defect rather than of the layouts.

**The shared layout is faster in every regime measured, and far more stable.**
Cold and warm times for a 1,000-query batch against DinoLedge with a 4 GiB
budget, and a 2,000-query batch against `dino_dog_toy` with a 256 MiB budget:

| Chunk target | DinoLedge tree-local | DinoLedge shared | `dino` tree-local | `dino` shared |
|---|---|---|---|---|
| 256 KiB | 3.23 s / 0.04 s | 2.56 s / 0.08 s | 1.36 s / 1.14 s | 0.30 s / 0.10 s |
| 1 MiB | 4.37 s / 0.04 s | 2.44 s / 0.08 s | 3.17 s / 2.99 s | 0.27 s / 0.10 s |
| 4 MiB | 10.01 s / 6.38 s | 5.17 s / 0.08 s | 14.95 s / 15.05 s | 0.27 s / 0.10 s |
| 8 MiB | 14.97 s / 11.43 s | 3.63 s / 0.08 s | 29.11 s / 29.63 s | 0.24 s / 0.10 s |
| 16 MiB | 25.19 s / 21.59 s | 3.63 s / 0.07 s | 57.99 s / 58.48 s | 0.25 s / 0.10 s |

The shared columns barely move. The tree-local columns vary 43x on `dino_dog_toy`
across the same chunk range, because four vector copies do not fit the budget and
larger chunks make each eviction cost more to undo. Tree-local's one win is a
fully resident warm batch — 0.04 s against 0.08 s on DinoLedge — where it scans a
leaf's descriptors in place while the shared path resolves them one at a time.
Both are under a tenth of a second, and it is the only cell where tree-local
leads.

**Shared reaches its plateau on a much smaller budget**, which is the same
advantage seen from the other side. On `dino_dog_toy`, shared is at full speed by
256 MiB (0.24 s) while tree-local still needs 1 GiB (0.48 s) and takes 8.03 s at
64 MiB. A file 3.3x smaller fits a cache 3.3x sooner.

**Descriptor block reuse is a tree-0 effect, and it collapses as trees are
added.** The shared corpus is laid out in tree-0 leaf order, so tree 0's leaves are
contiguous and share blocks; every other randomized tree partitions the corpus
differently and contributes scatter. Measured on `dino_dog_toy` with 4 KiB blocks
(32 rows), one cold query, varying only the tree count — descriptor accesses equal
the check count exactly, one block lookup per check, so a miss is a block read and
the remainder is reuse:

| Trees | Checks | Blocks read | Checks per block | Block reuse |
|-------|--------|------------|-----------------|-------------|
| 1 | 136 | 15 | 9.07 | 89% |
| 2 | 131 | 38 | 3.45 | 71% |
| 4 | 130 | 90 | 1.44 | 31% |
| 8 | 135 | 103 | 1.31 | 24% |

At eight trees nearly every check needs its own block. It is tempting to expect the
opposite — the trees converge on the same near neighbours, so their leaves should
hit the same blocks repeatedly — and the convergence is real, but it never becomes a
block access: the per-query `checked` set means a descriptor evaluated by one tree
is not re-evaluated by another. The overlap is absorbed as *fewer checks*, and what
remains to be read is exactly the non-overlapping part, which scatters.

This is the quantity behind two other results. It is why small blocks win — when
scatter dominates, most of a large block is waste — and it is why a 1,000-query
batch touches 87% of a DinoLedge file's entries at 64 KiB blocks: 130 scattered
draws per query against 18,950 blocks reaches almost all of them.

There is headroom here. At four trees, ~130 checks span roughly eight leaves, so an
assignment that kept each leaf's members together would need on the order of ten
block reads rather than 90. No single assignment can do that for all T trees at
once, but nothing in the format prevents trying: the storage-row map is an explicit
permutation a writer may choose freely, so this is a writer policy question with no
wire-format consequence.

**Shared block size trades cold query time against file size, and the balance
sits far smaller than it first appeared.** Once descriptor blocks stopped being one
ZIP entry each, block size became free in entry count, which had been the whole
penalty for small blocks. What remains is a genuine two-sided tradeoff, measured on
DinoLedge with a 4 GiB budget and a 1,000-query batch:

| Block | Rows | File | Open | Cold batch | Decoded | Seed amp. |
|-------|------|------|------|-----------|---------|-----------|
| 2 KiB | 16 | 1,247.8 MB | 239 ms | 1.18 s | 466 MiB | 76x |
| 4 KiB | 32 | 1,228.4 MB | 165 ms | 1.45 s | 602 MiB | 89x |
| 8 KiB | 64 | 1,218.5 MB | 145 ms | 1.60 s | 816 MiB | 114x |
| 16 KiB | 128 | 1,213.8 MB | 130 ms | 1.86 s | 1,091 MiB | 164x |
| 64 KiB | 512 | 1,211.6 MB | 113 ms | 2.10 s | 1,458 MiB | 437x |
| 256 KiB | 2,048 | 1,211.9 MB | 139 ms | 2.04 s | 1,407 MiB | 1,407x |

Smaller blocks waste less per descriptor fetched — 466 MiB decoded at 2 KiB
against 1,458 MiB at 64 KiB for the same answers — and that dominates the extra
reads, because a read is now a seek to an offset rather than a directory lookup.
Against it, small frames compress worse (34 MB more file at 2 KiB than at 16 KiB,
mostly lost zstd context rather than frame headers) and the offsets array grows
enough to show up at open.

A warm batch is 0.07-0.08 s at every size, so none of this matters to a workload
that revisits its corpus; it is entirely about cold and one-shot queries.

**Chunk size still drives cold-start amplification steeply**, tree-local 96x to
4,373x on `dino_dog_toy` across 256 KiB to 16 MiB, because seeding a four-tree
search costs four independent subtree misses whatever the query. Shared is far
flatter, 221x to 576x, since its tree chunks carry no vectors.

**More query workers helped no corpus measured.** Per-query work is
sub-millisecond at these budgets, so rayon's per-task overhead dominates:
`dino_dog_toy` shared goes 0.24 s to 0.36 s from one worker to eight. The default
of one worker stands.

**Recall and results are unaffected by storage, as designed.** Every cell returned
neighbors and distances identical to the in-memory forest, and recall@1 against
exhaustive search was identical across layouts within a corpus — 0.433, 0.557 and
0.657 at a 128-leaf budget.

### What this suggests as defaults

On this evidence the shared layout is the better default: 3.3x smaller, faster or
equal in every regime, insensitive to a chunk-size choice that swings tree-local
by 43x, and at full speed on a budget a third the size.

For descriptor blocks the knee is around **4 to 8 KiB**, which is where most of
the cold-query gain has been taken and the file has grown by well under 1%. Going
to 2 KiB buys another 0.27 s on a cold batch for 2.8% more file and twice the open
latency, which is the right trade only for a corpus queried once. A 1 MiB chunk
target and one query worker remain reasonable.

Note this recommendation moved twice under measurement, both times because
something unrelated to the layouts was dominating: first a cache whose hit cost
scaled with its size, then one ZIP directory record per descriptor block. The
figure to distrust in future is any block-size guidance that has not been
re-derived since the last change to how a block is addressed.

The API keeps requiring an explicit choice regardless. Tree-local's resident warm
case is genuinely faster, the margin is a property of the corpus and the budget
rather than a constant, and a caller who knows their working set fits in memory
has a real reason to pick it.

### What this does not settle

These runs cover one machine, one filesystem, `uint8` descriptors and four trees.
The twenty-tree case, where a shared corpus avoids nineteen copies rather than
three and the size argument is strongest, is unmeasured. So is `float32`, which
the format carries and the Python bindings do not expose.

The shared layout still resolves descriptors one at a time, at roughly twice
tree-local's cache-hit count for identical work; returning a borrow under the
existing pin and grouping a leaf's reads by block would close the one cell where
tree-local leads. That remains proposed in
[drafts/kdf-shared-descriptor-reads.md](../../drafts/kdf-shared-descriptor-reads.md),
now as the only outstanding half.

## Acceptance checks

### DinoLedge packing example

The [format case study](../../formats/kdf-file-format.md#dinoledge-case-study-2026-09-09)
measures 9,702,948 real 128-D descriptors across 1,196 images. For the
four-tree/16-feature-leaf layout, calculated counts are:

| Target decoded size | Subtree chunks per tree | Features per subtree | Actual subtree size | All ZIP entries, including source mapping |
|---------------------|-------------------------|--------------------|---------------------|-------------------------------------------|
| 1 MiB | 2,048 | 4,737–4,738 | 0.629 MiB | 32,940 |
| 4 MiB | 512 | 18,951–18,952 | 2.515 MiB | 8,364 |
| 8 MiB | 256 | 37,902–37,903 | 5.029 MiB | 4,268 |
| 16 MiB | 128 | 75,804–75,805 | 10.058 MiB | 2,220 |

Each tree also has one routing chunk in all four cases. The routing working
set across four trees is only 270,204 bytes at the 1 MiB target, 67,452 at
4 MiB, 33,660 at 8 MiB, and 16,764 at 16 MiB. These are decoded array sizes.
The root chunks can naturally remain hot without pinning entire trees.

Seeding a cold four-tree query touches one subtree per tree plus routing:
approximately 2.77 MB decoded at the 1 MiB target versus 42.2 MB at 16 MiB,
before best-bin-first expansion. Under the descriptor compression proxy, the
vector portions of those four subtree reads alone would be roughly 1.87 MB
versus 29.9 MB compressed. These are structural estimates, not measured query
I/O or latency. Four cold chunks do not imply four physical reads: each has
three or four entries, optionally coalesced, and the OS adds its own caching/read-ahead.

A 256 MiB decoded cache fits roughly 406 of the small subtree chunks after
routing arrays, versus about 25 of the 16 MiB-target subtrees; actual capacity
is lower with allocation overhead and cached origins. It holds about 5% of the
total 5.4 GB decoded forest either way. This is a useful corpus for exercising
eviction under an explicit cache budget, but whether it exceeds machine RAM
depends on the benchmark host. No query timing or recall benchmark was run.

This example supports starting with smaller chunks for cold-query experiments;
it does not establish the best size for warm or locality-ordered batches. Its
descriptor-only compression probe changed little between chunk sizes, suggesting
I/O amplification and reuse should be measured before prioritizing compression
ratio. Actual tree ordering still needs to be tested. See the companion case
study for the full size budget and source inventory fingerprint.

### Descriptor deduplication estimate: four versus twenty trees

This is a cost model for the DinoLedge inventory above, not a measured query
benchmark. Deduplication here means storing each corpus feature's descriptor
once across trees, not merging distinct feature IDs with identical bytes.
Keep the exact same trees, traversal and check budget, so recall does not change.
Cross-tree distance evaluations are already deduplicated in the existing search;
sharing stored descriptors does not save those computations a second time.

The shared-layout model stores vectors in tree-0 leaf order, retaining 2,048
descriptor blocks matching the 1 MiB-target subtree partition. Each block has
4,737–4,738 vectors (about 0.579 MiB decoded). Trees retain their nodes and
feature-ID lists, with descriptors removed. A single original-feature-ID to
shared-storage-row uint32 map costs 38,811,792 bytes; this model keeps it resident.
The inverse permutation is not needed for ANN because leaves already carry
original feature IDs. This extra map avoids storing a second address in every
tree leaf. Other addressing designs are possible and must declare their costs.

| Storage estimate, decimal GB | 4 trees | 20 trees |
|--------------------------------|---------|----------|
| Duplicated decoded trees/vectors | 5.400 | 27.000 |
| Shared decoded trees/vectors plus address map | 1.713 | 3.441 |
| Decoded reduction | 68% | 87% |
| Compressed descriptor copies removed, at proxy ratio 0.772 | 2.876 | 18.217 |
| Duplicated compressed planning range | 3.9–4.3 | 19.5–21.5 |
| Shared compressed planning range | 1.0–1.45 | 1.0–3.2 |

The wide shared-file ranges reflect unmeasured compression of tree node and ID
arrays, especially for twenty trees; the lower ends approximate the descriptor
floor, not an expected complete file size. Upper planning allowances keep these
arrays and the address map at decoded size, then allow small metadata overhead.
Origin mappings are common to both designs. The 20-tree duplicated range scales
the earlier four-tree estimate; it is not a newly measured compression result.

**Cold-query model.** DinoLedge leaves average 9.253 descriptors. Seeding visits
one leaf per tree before checking the work budget. Assume no repeated feature
IDs across these leaves for this calculation, a resident address map, and an
initially empty chunk cache. In tree 0, the leaf requires one shared descriptor
block. Model each descriptor in other trees as an independent uniform draw
among B = 2,048 blocks. This is a deliberately scattered-access scenario, not
a claim about the real trees: correlated trees may have substantially better
locality. If q = (T - 1)*9.253, the expected total distinct descriptor blocks is
`1 + (B - 1)*(1 - (1 - 1/B)^q)`.

| Seeding only, 1 MiB target | 4 trees | 20 trees |
|--------------------------|---------|----------|
| Leaf member visits, before dedup | about 37 | about 185 |
| Duplicated subtree blocks fetched | 4 | 20 |
| Shared descriptor blocks, scattered model | about 29 | about 169 |
| Duplicated decoded bytes, including routing | 2.77 MiB | 13.86 MiB |
| Shared decoded descriptor bytes, excluding tree reads | 16.52 MiB | 97.99 MiB |
| Descriptor-only byte amplification versus duplicated vectors | 7.1x | 8.5x |

Shared queries additionally read tree nodes and feature-ID lists. If their
partitions are retained, those initial tree arrays total about 0.46 MiB for
four trees and 2.30 MiB for twenty. Thus total decoded seeding traffic in this
model is about **6.1x / 7.2x** the duplicated baseline. Descriptor compression
is similar in the proxy measurements, so compressed descriptor traffic has
roughly the same amplification. Small separate tree entries add read overhead.
Under ideal cross-tree locality, shared descriptor blocks could instead number
one to T for seeding: sharing could match or improve on the duplicated bytes.
Neither scenario predicts actual recall, latency or whole-query block counts.

At checks=128, four-tree seeding normally leaves more best-bin-first work, while
twenty-tree seeding can already exceed 128 (about 185 distinct evaluations if
there is no cross-tree overlap). Real cross-tree duplicate IDs reduce both
checks and shared descriptor requests. Consequently the table cannot be scaled
to a whole query by assuming identical numbers of leaf visits for 4 and 20 trees.
Later visited leaves may reuse an already loaded subtree in either layout.

If a query is dominated by cold reads and decoding, the scattered model suggests
roughly **6–8x more transfer/decode work**, with potentially additional latency
from more dependent reads. It does not establish a 6–8x wall-clock slowdown:
coalescing, storage latency, node caching and cross-tree locality all matter.
No SSD bandwidth or decompression throughput has been measured for this case.

**Warm cache and batches.** A shared descriptor corpus plus its address map is
about 1.28 GB decoded, independent of tree count. Once resident, descriptor disk
reads and zstd decoding disappear. A budget sufficient for all shared data is
about 1.71 GB for four trees or 3.44 GB for twenty (plus common origins, metadata
and runtime overhead), versus 5.40 or 27.00 GB for the duplicated design. In that
budget interval, sharing can win substantially even if its first query loses.
With both forests completely resident, expect similar arithmetic work: shared
storage adds address indirection but reduces memory footprint, while duplicated
storage provides contiguous leaf scans. A measured cache/CPU benchmark is needed
to determine which wins. First full load and export also process far fewer vector
bytes with sharing, though their total speedup is not the storage reduction ratio.

At only a 256 MiB cache, neither design holds the corpus; uniform shared accesses
may still have low hit rates. Sharing improves reuse across trees and queries,
but does not guarantee a warm-cache speedup at that budget. Locality-ordered
batches are the important favorable case to measure.

**Design implication.** Benchmark shared descriptors with their own smaller block
size, independent of tree chunks, before accepting T-fold duplication. A 16 KiB
descriptor block holds 128 SIFT rows; even 185 distinct cold block requests decode
only 2.9 MiB of descriptor data, although they incur many more reads and ZIP
entries. Compare 16, 64, 256 KiB and the subtree-sized baseline, accounting for
the address map, directory overhead and tree bytes. The tradeoff is particularly
compelling at twenty trees: roughly 18 GB of avoided compressed copies gives
considerable room to trade some cold-query latency for capacity and warm reuse.
Both layouts are part of the version-1 wire contract, so these measurements
choose a default rather than deciding whether sharing is supported at all.

### How parity is tested

Parity is the property that carries the risk here, so the tests assert it
against the in-memory forest rather than against recorded expectations. Each
case builds a forest, exports it, opens the file, and compares. Chunk and block
targets are set to a few hundred bytes so that a forest of a few dozen
descriptors still spans many chunks: the traversal that matters is the one that
crosses a chunk boundary, and a realistic target would put the whole fixture in
one chunk and test nothing.

[`persistent.rs`](../../../crates/sfmtool-core/src/features/kdforest/persistent.rs)
holds three cases. The `u8` case exports a four-tree forest in both layouts and
compares eleven queries at leaf budgets of 0, 1, 7, 31 and 1000, plus a batch
call. It asserts equal leaf-check counts as well as equal neighbors, because
matching neighbor IDs alone would not catch a file-backed traversal that visits
a different set of leaves and agrees by luck — the counts are what pin the
far-child queue order, and with it the logical-ID tie-break. The `f32` case
covers signed-zero routing at a split plane, an infinite cutoff, and the two
rejected queries (NaN coordinate, negative `max_dist`). The concurrency case
gives eight threads one identical query against a cache smaller than the file,
so every load races, evicts and is deduplicated; it asserts that opening a
shared-layout file reads no descriptor block at all.

[`sfmtool-kdf-format/src/tests.rs`](../../../crates/sfmtool-kdf-format/src/tests.rs)
covers the format in isolation: a round trip in both layouts down to node
addresses and logical IDs, a full `verify_kdf` pass, SIFT origins resolved
lazily and in the caller's requested order including repeats, the writer's
refusal to overwrite an existing destination, and a `ResourceLimit` when the
address-map budget cannot hold the shared row map. The corruption case is the
one that pins the laziness contract: a damaged shared descriptor block leaves
`open` succeeding and reading nothing, makes the direct access fail, and makes
full verification fail.

The negative surface these do not reach — malformed node references, cycles,
non-contiguous leaf ranges, invalid permutations, wrong entry sets, unsupported
versions, truncated frames, and `verify_sift_sources` against real `.sift` files
— is proposed in
[drafts/kdf-validation-tests.md](../../drafts/kdf-validation-tests.md).

## The Python surface

The benchmark plan above is a Python job — a sweep over corpora, layouts, chunk
sizes and cache budgets, reporting latency percentiles and recall — so the
`uint8` path is bound on the `sfmtool.spatial` submodule, in
[`spatial/kdf.rs`](../../../crates/sfmtool-py/src/spatial/kdf.rs), beside the
in-memory `KdForest` it is measured against.

```python
from sfmtool._sfmtool.spatial import (
    KdForest, LazyKdForest, write_kdf, kdf_file_summary, verify_kdf,
)

forest = KdForest(descriptors, num_trees=4, leaf_size=16, seed=7)
write_kdf(forest, "corpus.kdf", layout="shared", descriptor_block_bytes=64 << 10)

lazy = LazyKdForest("corpus.kdf", cache_bytes=256 << 20, query_workers=4)
indices, distances, stats = lazy.query_with_stats(queries, k=2, max_leaf_checks=128)
io = lazy.io_stats()
amplification = io["decoded_bytes"] / (stats["checks"] * lazy.dim)
```

Three things about that surface follow from what it is for rather than from
the Rust API it wraps.

`layout` has no default. Which layout to ship is the open question, so a
caller states one, and `descriptor_block_bytes` is *rejected* for the
tree-local layout rather than ignored — an argument silently dropped would
make two cells of a sweep run identically under different labels.

`reset_io_stats` exists because the alternative for separating an open from
the queries after it, or a cold pass from a warm one, is reopening the file,
and reopening also drops the cache. The counters zero; what the cache holds
does not.

`kdf_file_summary` splits a file per role rather than reporting one total,
reading only the ZIP central directory and the metadata entry, so it costs the
same on a 5 GB file as on a 5 KB one. `tree_vectors` is what the shared layout
removes T-1 copies of; `shared_vectors`, `shared_row_map` and
`shared_block_offsets` are what it adds back; `tree_chunks` — a chunk's node,
split and feature-ID arrays, which share one entry — is byte-identical either way,
which is how a reader can tell a size comparison is comparing one forest stored
twice rather than two different forests.

Errors are split by what a sweep must do about them: a budget that cannot hold
what was asked for raises `MemoryError` (try another cell), a malformed or
damaged file raises `OSError` (stop), and a bad argument raises `ValueError`.

## Out of scope

There are no updates or appends to an existing file, no remote HTTP reads, no
CLI, no automatic precision calibration, no alternate metrics, and no external
descriptor dependencies. Existing in-memory query callers keep their current
API. The Python bindings cover `uint8` only.

The format and this query path both cover `u8` and `f32`, matching the scalar
types the in-memory forest already supports. The open questions that remain are
measurements, not design: whether T-fold vector duplication is worth its cost
against a shared corpus, and where the chunk-size and cache-budget defaults
land. Both are settled by the benchmark plan above without a format change.
