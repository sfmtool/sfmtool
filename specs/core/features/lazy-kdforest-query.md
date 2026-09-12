# Lazy KD-Forest Queries

A persistent kd-tree forest lets a process search a large set of descriptors
while keeping only the portions it visits in memory. The file-backed query path
in core does this over chunked trees with either leaf-local vectors or a shared
vector table, targeting local seekable files, repeated queries, and corpora
larger than the configured memory cache. It preserves the in-memory forest's
search behavior in both layouts.

Packing and cache values are configurable. The interface lists provisional
defaults; the measurements and their scope are documented in
[Current access path and performance diagnosis](#current-access-path-and-performance-diagnosis).

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

```rust
pub fn verify_kdf<S: KdfScalar>(path: &Path, options: LazyKdForestOptions)
    -> Result<Verification, KdfError>;
pub fn verify_sift_sources(path: &Path, options: LazyKdForestOptions)
    -> Result<Verification, KdfError>;
```

`verify_kdf` checks the self-contained archive. `verify_sift_sources` is the
explicit, separate audit against the live workspace recorded in its provenance.
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

Per-query dedup uses reusable bitsets for feature IDs and logical node IDs, with
lists of touched words to clear between queries. Scratch includes bits proportional
to corpus and tree size per Rayon job. Queue memory grows with explored branches. Batch output is O(M*k);
metadata, scratch, output, compressed buffers and decoder workspace are additional
to the cache. Checked arithmetic and configurable worker count constrain growth;
this is bounded residency of file data, not constant total memory for arbitrary k.

Treat open files as immutable for the handle lifetime. Atomic replacement can
leave an old handle serving its old snapshot; in-place writes are unsupported.
Chunk validation checks references before dereference and detects revisited
logical nodes per query to prevent malformed cycles. Full semantic verification
is an explicit offline operation, never implicit at lazy open.

## Where this sits in the literature

[Beis & Lowe 1997](https://www.cs.ubc.ca/~lowe/papers/cvpr97.pdf) describes
Best-Bin-First search with a bounded search effort.
[Lowe 2004](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf), ? 7.2, applies it to
SIFT matching with a cutoff of 200 candidate checks.
[Muja & Lowe 2009](https://www.cs.ubc.ca/~lowe/papers/09muja.pdf) describes
randomized kd-trees searched through a shared priority queue. These are the
algorithmic references for this implementation; they do not evaluate its
compressed, file-backed access path.

[Muja & Lowe 2014](https://www.cs.ubc.ca/~lowe/papers/14mujaPAMI.pdf), ? 5,
considers disk-backed data among the options for corpora larger than memory and
implements distributed nearest-neighbor search across machines. The benchmarks
below evaluate a different configuration: one machine, a local file, and a bounded
decoded cache. They do not compare against distributed FLANN or establish how
storage hardware changes that comparison.

The recall metrics also differ. Lowe reports loss of correct matches after the
ratio test on a 100,000-keypoint database; the measurements below include raw
recall@1 on larger corpora. Raw recall alone does not establish how many usable
matches this pipeline loses. That requires measuring the downstream matcher.

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

## Current access path and performance diagnosis

[`persistent.rs`](../../../crates/sfmtool-core/src/features/kdforest/persistent.rs)
borrows one tree chunk while following nodes within it. Tree-local leaf vectors
are evaluated in place. Shared leaves copy unchecked IDs into reused scratch,
release the tree pin, and borrow consecutive descriptors from each block. A query
holds no pin while admitting another block. Descriptor encounter order, queue
priorities and check counts are unchanged, including equal-distance ties.

[`cache.rs`](../../../crates/sfmtool-kdf-format/src/cache.rs) maintains an indexed
doubly linked LRU list per shard. Hits and victim removal take constant time;
eviction walks only older pinned entries before an available victim. Recency slots
are reused. Pins borrow the cache owner rather than modifying a global reference
count; an `Arc` still pins each value and its bytes count against residency. A
miss-only global gate enforces the total in-flight decode limit without reducing
cache sharding. Shard count is capped by the largest validated item in the file,
not the caller's permissive `max_chunk_bytes` ceiling. Tree IDs contribute low
bits to shard selection, so different trees' root chunks can use different locks. Pin-release notifications synchronize with the admission mutex,
and waiters register before checking capacity.

Eager reload validates the reconstructed graph for cycles, duplicate/missing
features and unreachable nodes before handing it to the unchecked in-memory
traversal. Provenance includes the default check budget, which reload restores;
older files without this field retain the balanced fallback.

Batch queries write directly into preallocated result rows. Ordered batches and
self-joins restore row order by cycling the permutation in place, avoiding a
second `N * k` result table and one neighbor-vector allocation per query. The
permutation requires O(N) IDs, and the clustering stage still has its own
O(N * k) candidate arrays.

There are two distinct sources of overhead relative to an eager forest:

- With a resident working set, lazy traversal validates addresses, tracks visited
  nodes, resolves storage rows and acquires pins. Eager traversal indexes resident
  arrays directly. Disk speed cannot explain a run with zero reads.
- Under cache pressure, misses require admission, reads, decompression and integrity
  checking. Eviction can make the same block decode repeatedly. The old eviction
  scan additionally cost CPU time proportional to resident entries per replacement.

A release cache-only probe (`benchmark_cache_churn`, 10,000 replacements, no file
or zstd work) on 2026-09-10 measured:

| Resident entries in one shard | Previous ns/replacement | Indexed LRU ns/replacement |
|---:|---:|---:|
| 1,024 | 8,197 | 457 |
| 16,384 | 136,177 | 346 |
| 65,536 | 1,226,745 | 391 |

These are diagnostic samples, not disk throughput. The existing 256-query synthetic
resident benchmark measures tree-local at 6.31 ms before and 0.74 ms after, and
shared at 8.68 ms before and 2.29 ms after. Both lazy runs use four workers. Eager
with four workers measures 0.27 ms; the earlier benchmark used the machine-wide
pool for eager, so its earlier eager number is not a worker-matched comparison.
The current benchmark asserts indices and distances, prints traversal/I/O counters,
and accepts `KDF_BENCH_WORKERS` for both paths. The batch performs 34,157 checks
and no warm reads. Current cache-hit counts are 4,671 tree-local and 30,105 shared.
Shared-access overhead remains even when file I/O is absent.

The shared corpus uses explicit-offset reads: `read_at` on Unix and `seek_read`
on Windows, with retries for interrupted/short reads and an error for premature
EOF. The handle's cursor is never used to choose a descriptor frame. This removes
the mutex around seek/read and uses one positioned read rather than two separate
operations. Each executing thread reuses a zstd decompressor context; decoded
output remains subject to the existing exact-size and hash checks. The context's
workspace is additional to decoded-cache residency and lasts with that thread.

On Windows, positioned reads on a single synchronous handle still serialize.
The corpus therefore opens a bounded pool of independent handles, sized by
`query_workers`, and assigns executing threads to stable slots. It uses
[ReOpenFile](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-reopenfile)
on the existing handle, preserving the original file-system object if the path is
atomically replaced. Duplicating a handle with `try_clone` would share synchronous
I/O state. Unix uses `read_at` on the original snapshot without a handle pool.
The Windows regression test checks both snapshot identity and independent cursors.

A four-thread diagnostic on 2026-09-11 measured 7.02 microseconds per completed
block with a shared synchronous handle and no cache, 3.52 microseconds with the
corrected cache and independent handle pool, and 3.13 microseconds with private
handles and no cache. These are OS-warm throughput measurements, not device
latency or an end-to-end speedup.

A separate one-thread diagnostic on a 547,651-descriptor file (32 rows per block,
20,000 sampled frames, OS-warm reads) measures the stages independently:

| Stage | Mean microseconds/block |
|---|---:|
| Seek, read and frame allocation | 4.04 |
| Explicit-offset read and frame allocation | 1.12 |
| New zstd context and decode | 4.08 |
| Reused zstd context and decode | 3.76 |
| Hash comparison and decoded-byte copy | 0.50 |

These calls are timed separately and do not include cache admission, traversal or
multiworker contention. The positioned-read measurement follows the ordinary read
of the same frame, so it describes OS-warm access, not physical device latency.
Run the ignored `profile_corpus_misses` release test with `KDF_PROFILE_PATH` set to
a shared u8 file to reproduce the decomposition. The test checks both read methods
and both decoder methods produce identical bytes.

### End-to-end held-out images (2026-09-11)

The shared-layout image-query benchmark indexes 547,651 descriptors from 82
Dino images and queries three held-out images (23,238 descriptors total). It uses
four workers for eager and lazy, four trees, 16-feature leaves, k = 11, check
budget 128, 4 KiB descriptor blocks and 64 KiB tree chunks. Every lazy and reloaded
eager result is checked against the in-memory indices and distances.

| Decoded cache | Before: later image seconds | After: later image seconds |
|---|---:|---:|
| 16 MiB | 4.72?5.23 | 3.13?3.16 |
| 64 MiB | 1.49?1.51 | 1.23?1.23 |
| 256 MiB | 0.37?0.38 | 0.10?0.13 |

Each endpoint is a run's median over the second and third images, across two
runs per implementation. The baseline is the preserved pre-review installed
release extension, not a fresh rebuild of the PR head. The final runs bracket
one baseline run. OS page cache was not flushed. First-image lazy times after the
fix are 2.48?2.54 s, 1.04?1.12 s and 0.25?0.26 s respectively; ?cold? refers only
to the decoded cache. [Raw measurements and command](kdf-review-2026-09-11.json)
include all per-image times and available I/O counters.

Eager traversal still takes roughly 0.02 s per later image. At 16 MiB, the final
image causes approximately 632,700 block reads: 1.99 GB of compressed input and
2.54 GB of decoded output from a 73 MB file. Repeated eviction and decoding
explain why a bounded lazy cache remains much slower. At 256 MiB the same image
needs one read and no eviction; the remaining gap is traversal validation,
address mapping and pin/cache bookkeeping. The fixes reduce those costs without
changing candidate order, but do not make the two access paths equally cheap.

The historical measurements below predate this access-path revision. They remain
records of those runs, not predictions for the current implementation.

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

**The hash directory grows with the block count, and is read at open.**
`descriptor_blocks_xxh128` holds one digest per descriptor block, so a small block
size makes a large directory: 9.7M descriptors in 4 KiB blocks is ~303,000 digests,
around 10 MB of JSON parsed before the first query, and 2 KiB blocks double it.
This is the part of a small block size that is *not* free — the container made block
count free in ZIP entries, but not here — and it is most of why open time climbs as
blocks shrink, the offsets array being the smaller term.

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

Two alternative orderings were measured against it and neither is worth adopting.
The storage-row map is an explicit permutation a writer may choose freely, so this
is a writer policy question with no wire-format consequence, and
[`scripts/kdf_descriptor_orderings.py`](../../../scripts/kdf_descriptor_orderings.py)
compares policies on one forest. At four trees and 4 KiB blocks, one cold query:

| Policy | `dino` blocks | `dino` reuse | DinoLedge blocks | DinoLedge reuse | DinoLedge batch reads |
|--------|--------------|-------------|-----------------|----------------|----------------------|
| tree-0 leaf order | 90 | 30.8% | 114 | 14.9% | 83,369 |
| Morton over 3 principal components | 123 | 5.4% | 133 | 0.7% | 110,409 |
| greedy co-occurrence packing | 74 | 43.1% | 111 | 17.2% | 86,551 |

A **projection order** — descriptor-space locality via a Z-order curve over the top
principal components — is clearly *worse*, and the reasoning that motivated it was
wrong. Every tree's leaf is spatially compact, so an order preserving descriptor-space
locality ought to serve all T trees at once. But three principal components are far
too lossy a summary of 128 correlated dimensions: descriptors adjacent on the curve
are frequently not leaf-mates in any tree, so the order gives up the one thing the
default reliably has — tree 0's leaves exactly contiguous — and buys almost nothing
back. Reuse falls to 0.7% at 9.7M descriptors and a batch reads 32% more.

**Greedy co-occurrence packing** — walking leaves round-robin across trees and
emitting each leaf's not-yet-placed members together — does beat the default on the
metric it targets, 74 blocks against 90 at 694k descriptors. The advantage nearly
vanishes at 9.7M (111 against 114) and reverses on batch reads. It is not a win.

These measurements compare three orderings; they do not establish a limit on
what other packing methods can achieve. At 694k descriptors all three policies
read about 22,200 blocks from a corpus of 21,698 blocks. These runs do not show a
useful batch-read reduction.

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

**Leaf size 32–64 is the measured file-backed range.** A leaf's members are
contiguous in the stored corpus, so larger leaves place more of them in one block
and need smaller tree arrays. They also examine more descriptors per visited
neighbourhood. Because leaf size changes the index,
[`scripts/kdf_leaf_size.py`](../../../scripts/kdf_leaf_size.py) finds the smallest
integer check budget reaching the recall target for each forest before timing it.
The earlier power-of-two budget grid overshot that target unevenly and made large
leaves appear more expensive than an equal-recall comparison supports.

The optimized reader was measured on 570,889 `dino_dog_toy` descriptors, with
1,000 descriptors held out, four trees, four workers, 4 KiB descriptor blocks,
64 KiB tree chunks and recall@1 >= 0.65. The table reports the median of three
independent forest/query seeds; each seed's time is itself the median of three
fresh-reader batches:

| Leaf | Exact budget range | File | 64 MiB batch | 256 MiB batch |
|---:|---:|---:|---:|---:|
| 8 | 157–182 | 80.5 MB | 270 ms | 194 ms |
| 16 | 190–221 | 76.4 MB | 234 ms | 155 ms |
| 32 | 241–278 | 73.6 MB | **210 ms** | 135 ms |
| 64 | 345–415 | 72.3 MB | 218 ms | 138 ms |
| 128 | 524–548 | **71.4 MB** | 217 ms | **132 ms** |

At 64 MiB, leaf 32 is fastest and leaf 64 is within 4%; at 256 MiB, leaves
32–128 differ by 6 ms. Leaf 8 is consistently slower. Leaf 128 saves only 1.2%
of file size over leaf 64 and has the widest pressured-cache timing range, so 64
is the useful upper end rather than 128.

Whole-image and patch-constellation holdouts expose a cache-dependent crossover.
These compare leaf 16 at budget 210 with leaf 64 at budget 384, using the
seed-zero calibrated budgets and rounding leaf 64 up from 383. Every lazy
neighbour index and distance equals its in-memory forest reference:

| Workload | Cache | Leaf 16 | Leaf 64 |
|---|---:|---:|---:|
| Held-out image, later-image median | 16 MiB | **5.12 s** | 5.92 s |
| Held-out image, later-image median | 64 MiB | 1.81 s | **1.42 s** |
| Held-out image, later-image median | 256 MiB | **0.169 s** | 0.265 s |
| Patch constellation, median query | 16 MiB | **231 ms** | 353 ms |
| Patch constellation, median query | 64 MiB | 100 ms | **75 ms** |
| Patch constellation, median query | 256 MiB | **14 ms** | 19 ms |

At 64 MiB, leaf 64's smaller file and tree working set reduce reads enough to
outweigh its larger check budget. At 16 MiB, that budget produces more repeated
misses; with the whole file resident, the additional distance work is exposed.

A 256 MiB cache is resident for this 72–80 MB corpus, but pressured for
DinoLedge's 1.19–1.26 GB files. Two held-out DinoLedge images and four patch
constellations were therefore measured at the established 1,000-query calibration
points: leaf 16 at budget 128 (recall 0.657), leaf 32 at 256 (0.679) and leaf 64
at 512 (0.685). The larger leaves have slightly higher recall, so the table is a
practical bracket rather than an exact iso-recall comparison:

| Leaf | File | Held-out image | Reads/first image | Patch query | Reads/patch |
|---:|---:|---:|---:|---:|---:|
| 16 | 1,256 MB | **3.77 s** | 677,358 | **89 ms** | 14,636 |
| 32 | 1,210 MB | 6.09 s | 1,208,432 | 134 ms | 25,421 |
| 64 | **1,188 MB** | 11.45 s | 2,252,895 | 236 ms | 45,607 |

All lazy results equal their in-memory forest's indices and distances. With
256 MiB holding about one fifth of the file, reads scale with the check budget;
the 5% file reduction from leaf 16 to 64 does not offset four times as many checks.
The general forest default remains 16, and it is also the measured choice for a
large shared index with a 256 MiB cache. Leaf 32–64 remains useful when the cache
is near the complete stored working set or file size has more weight than query
latency.
[The recorded measurements](kdf-leaf-size-2026-09-11.json) include all three
small-corpus seeds and both corpora's holdout comparisons.

**Query workers pay off only once the cache is sharded.** Every node and every
descriptor a query touches takes a cache lock, so with one lock for the whole
cache the workers serialize on it and adding them does nothing. The cache is split
into up to 16 independently locked shards, chosen by the low bits of the cache key.
Interleaved arms, nine rounds of 20,000 queries against a fully resident
`dino_dog_toy` file, k = 11:

| Cache | Workers | Median | Range |
|-------|---------|--------|-------|
| 1 shard | 1 | 74.2 us | 61.9-86.0 |
| 16 shards | 1 | 72.4 us | 59.6-81.5 |
| 1 shard | 4 | 72.6 us | 69.1-76.9 |
| 16 shards | 4 | **46.8 us** | 45.8-49.1 |

In these runs, one-worker timings overlap. Four workers are 1.55x faster with
sharding than without it. This result describes this workload and configuration. Removing the
contention also tightens the spread to about +/-3%, against +/-15% for the
contended arms.

The shard count is not free to choose. Admission is per shard, so a shard smaller
than the largest item a caller may ask for could never admit it and the caller
would block forever; the count is capped at `cache_bytes / max_chunk_bytes`. A
cache only just large enough for one chunk collapses to a single shard, which is
the unsharded cache.

Keys are dense block and chunk indexes, so their low bits spread uniformly across
shards with no hashing. Sharding on the cached *contents* instead — a descriptor's
leading bytes, say — would be badly skewed, because SIFT descriptors carry many
small and zero components and most keys would land in a few shards.

The maps themselves are keyed with XXH3 rather than the standard library's
SipHash-1-3. SipHash exists to resist hash flooding from attacker-chosen keys;
these keys are dense integers this crate generates while walking a file it has
already validated, and the crate already hashes every stored section with XXH3, so
this adds no dependency and no second hash to justify.

An earlier version of this section attributed workers not helping to rayon's
per-task overhead on sub-millisecond queries. That was wrong — it was lock
contention, which is why sharding fixes it and task overhead would not.

**A note on measuring any of this.** Timings on this machine drift by up to 2x
between runs, enough to invent effects and hide real ones: during this work a
single sample suggested one change was worth 3.2x, repeated runs put the same
configuration at half that speed, and a third run put it back. Comparisons here
are therefore *interleaved* — every arm measured in every round, so drift is
shared rather than attributed to one arm — and reported as medians with ranges. A
timing claim in this file that is not measured that way should be distrusted.

**Recall and results are unaffected by storage, as designed.** Every cell returned
neighbors and distances identical to the in-memory forest, and recall@1 against
exhaustive search was identical across layouts within a corpus — 0.433, 0.557 and
0.657 at a 128-leaf budget.

**A cache lookup per descriptor is the wrong shape for a whole-corpus algorithm,
and tuning the cache does not change that.** One query at a 128-check budget makes
about 241 cache lookups — 132 for descriptors, the rest for nodes. The in-memory
forest answers the same query in about 1 us, because a resident leaf is a slice it
indexes directly and a descriptor comparison is a few nanoseconds. The file-backed
path, fully resident and with every fix below applied, takes about 47 us.

Four costs have come off that path, every one invisible in a profile of the
algorithm and obvious in the data structure. Promoting an entry on a hit walked an
ordered recency list, making a hit O(resident entries). It then hashed the key
twice, once to read the value and once inside the recency update. Dropping a pin
called `notify_all` unconditionally, waking nobody, once per node and per
descriptor examined. And one lock served every access, so workers could not run.
What remains after all four is a lock, a hash lookup and an `Arc` clone per
descriptor, against an in-memory path that does an indexed read — which is a
difference in kind, not in tuning.

For the sparse queries this path was designed around, 47 us is fine: the
alternative is not having the corpus at all. For an algorithm that queries *every*
descriptor it is not. The whole-corpus matcher in
[track-cluster-matching.md](track-cluster-matching.md) runs correctly against a
`.kdf`, returning byte-identical clusters at every scale measured, at tens of
microseconds per query against the in-memory path's ~1 us.

Reaching ~1 us means a descriptor access costing nanoseconds, which means no lock
and no hash lookup on the path: the corpus indexed directly, with the operating
system's page cache doing the caching. That is the memory-mapped flat-array design
[kdf-file-format.md](../../formats/kdf-file-format.md) records as the shape a
mapping consumer wants, and this measurement is the argument for it rather than
against.

**And an out-of-core index is not an out-of-core algorithm.** Cluster matching is
a self-join, so the file-backed path also supplies its own queries — it reads each
descriptor back out of the `.kdf` in stored order, uses it and drops it, and never
materializes the corpus. That works, and the clustering stage never looks at a
descriptor at all, only at neighbour indexes and distances. But the matcher's own
intermediates are `Theta(N * d)`: a neighbour table and a candidate array, each
`N * (d + 1)` entries, in both paths. At 9.7M descriptors that is roughly 850 MB
each against a 1.24 GB corpus, so taking the index out of core removes about a
third of the footprint and no more. Measured at 696k descriptors the two paths peak
within 7% of each other, most of which is the Python process floor. Making this
algorithm genuinely out-of-core needs those arrays streamed too, which is work on
the matcher rather than on the index.

### Two real access patterns, and which one this path suits

The measurements above vary storage against a fixed synthetic query set. These two
vary the *workload*, because the answer to "is a file-backed index worth it" turns
out to depend far more on that than on any packing choice. Both are measured by
[`scripts/kdf_new_image_query.py`](../../../scripts/kdf_new_image_query.py) and
[`scripts/kdf_patch_localize.py`](../../../scripts/kdf_patch_localize.py), and both
return results identical to the in-memory forest at every cache budget tried.

**Fitting a new image into an existing capture.** A capture is indexed, an image
arrives, and its descriptors need their neighbours. Ten images are withheld from
the index, spread through the sequence so each arrival's temporal neighbours are
still present — the situation a real arrival is in, where withholding a contiguous
run would measure the hardest case instead.

There are three ways to be ready to answer, not two. The index can be rebuilt from
the `.sift` corpus, loaded from a `.kdf` into the same in-memory structure, or
queried from the file without materializing it at all. At 9.6M descriptors and
1,186 images:

| Ready by | Time to first answer | Per image after |
|----------|---------------------|-----------------|
| Rebuilding from `.sift` | 13-19 s | 0.37 s |
| Loading the `.kdf` | 32 s | 0.36 s |
| Opening the `.kdf` | **0.22 s** | 0.57 s |

**Loading is slower than rebuilding**, which is worth stating because it is the
opposite of what a persisted index is supposed to buy. The build is not the
expensive part: median splits over 9.7M points take about 9 s, while loading has to
decompress the corpus and every tree array and then scatter 9.7M descriptors into
feature-ID order, which is a random write across a 1.24 GB array. Persisting the
forest is therefore not a way to start faster in memory — it is a way not to be in
memory at all.

That is what the third row is for. Opening is two orders of magnitude quicker than
either, so a `.kdf` wins outright for a handful of arrivals and loses once a rebuild
amortizes: the crossover is about three images at 630k descriptors and about fifty
at 9.6M. Reloading is exact — a forest read back from a file answers identically to
the one written, because the file stores its topology, leaf order and feature IDs
rather than the seed it was built from.

It does *not* save memory for this pattern, which was the surprise. **A whole image
is not a sparse query**: 8,192 descriptors at a 128-leaf budget make about a million
checks, which reach 279,658 of a DinoLedge file's ~303,000 descriptor blocks — 92%
of the corpus. The cache therefore has to hold nearly everything or it thrashes, and
the difference is not subtle: 256 MiB against that 1.2 GB file takes 24 s per image
with 730,000 evictions, while 4 GiB takes 0.47 s with none. A budget large enough to
be fast is a budget as large as the corpus.

**Localizing a patch.** Given a rectangle in one image, take the constellation of
features inside it, look each one up, group the hits by source image, and keep the
images whose correspondences survive RANSAC on an affine model. `k` is larger here —
32 rather than the matcher's 11 — because most of a constellation feature's nearest
neighbours belong to images that do not contain the patch, and RANSAC needs the right
image to be among the candidates at all.

This is the pattern the file-backed path suits, and the contrast with a whole image
is the reason. A 400-pixel patch is 100-1,400 features, so on the 696k corpus it
touches 2,784 blocks and decodes 15 MB — against 279,658 blocks for a whole image.

| Corpus | Query, in memory | Query, `.kdf` | RANSAC | Reads | Decoded |
|--------|-----------------|--------------|--------|-------|---------|
| 696k, 85 images | 1.4 ms | 32 ms | ~433 ms | 2,784 | 15 MB |
| 9.7M, 1,196 images | 7.3 ms | 280 ms | ~2 s | 11,605 | 93 MB |

The index query is ~7% of end-to-end work at 696k and a similar order at 9.7M;
geometric verification dominates both. So the file-backed path's per-query overhead,
which is 20-40x on the query alone, costs a few percent of the job — and buys an
index that need not be rebuilt or held.

Chunk size does not help this pattern, which is worth stating because it looks like
it should: a sparse query descends through tree chunks, so smaller chunks ought to
pull less. Measured from 128 KiB to 4 MiB, decoded bytes move from 19.7 to 21.7 MB
and query time not at all beyond noise. The reason is already in
[Packing policy](#packing-policy): shared-layout chunks are packed with weights that
include vector bytes the layout does not store, so a chunk underfills its target by
roughly 17x and the target is close to decorative there. What drives decoded volume
is how many distinct blocks scattered queries reach, not how big each one is.

### What this suggests as defaults

On this evidence the shared layout is the better default: 3.3x smaller, faster or
equal in every regime, insensitive to a chunk-size choice that swings tree-local
by 43x, and at full speed on a budget a third the size.

Leaf size **16** remains the general forest default and the measured choice when a
256 MiB cache is substantially smaller than the index. For a shared persistent
index whose cache is close to its stored working set, **32 to 64** is the measured
range. Leaf 64 is useful near that cache threshold; resident execution can favor
leaf 16 or 32 once the check budget is calibrated to equal recall.

For descriptor blocks the knee is around **4 to 8 KiB**, which is where most of
the cold-query gain has been taken and the file has grown by well under 1%. Going
to 2 KiB buys another 0.27 s on a cold batch for 2.8% more file and twice the open
latency, which is the right trade only for a corpus queried once. A 1 MiB chunk
target remains reasonable. Query worker count is an execution setting and should
be measured separately from the stored layout.

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

The access path borrows descriptors and reuses a pin for consecutive requests to
the same block. Grouping nonconsecutive requests by block remains an experiment
in [drafts/kdf-shared-descriptor-reads.md](../../drafts/kdf-shared-descriptor-reads.md).
It must preserve encounter-order ties and include sorting and replay costs in
measurements; a smaller cache-hit count alone does not establish an improvement.

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

[`sfmtool-kdf-format/src/validation_tests.rs`](../../../crates/sfmtool-kdf-format/src/validation_tests.rs)
builds the negative surface around a hash-aware archive mutator. It changes a
decoded entry, recomputes the affected section and whole-file digests independently
of the writer, and rewrites the archive. That makes malformed child references,
cycles and shared children, bad leaf ranges, reserved fields, invalid tree and
storage permutations, cross-tree vector differences, split violations, nonfinite
`f32` values, unsupported metadata, wrong entry sets, duplicate ZIP names and
truncated frames reach the validator each test names instead of stopping at an
unrelated integrity mismatch. It also asserts that an unvisited chunk stays unread
and a warm resident access performs no read or decode.

The Python binding test extracts SIFT once from the included 270x480 Seoul Bull
image and reuses that file for every `verify_sift_sources` case: matching and
relocated workspaces, a missing source, a changed identity, an out-of-range source
feature and a mismatched descriptor. The same missing-source case confirms ordinary
queries and embedded origin lookup remain available. Synthetic origin mutation in
the Rust suite covers out-of-range image IDs and duplicate source pairs without
another extraction.

## The Python surface

The benchmark plan above is a Python job — a sweep over corpora, layouts, chunk
sizes and cache budgets, reporting latency percentiles and recall — so the
`uint8` path is bound on the `sfmtool.spatial` submodule, in
[`spatial/kdf.rs`](../../../crates/sfmtool-py/src/spatial/kdf.rs), beside the
in-memory `KdForest` it is measured against.

```python
from sfmtool._sfmtool.spatial import (
    KdForest, LazyKdForest, write_kdf, kdf_file_summary,
    verify_kdf, verify_sift_sources,
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

`layout` has no default even though the measurements recommend shared storage.
A caller whose complete working set is resident can still benefit from tree-local,
so the caller states the trade explicitly. `descriptor_block_bytes` is *rejected*
for the tree-local layout rather than ignored — an argument silently dropped
would make two calls run identically under different labels.

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
types the in-memory forest already supports. Measurements recommend the shared
layout, 4–8 KiB descriptor blocks and a 1 MiB tree-chunk target for sparse queries;
the API retains an explicit layout choice because the resident warm case remains
a legitimate tree-local workload. The unmeasured twenty-tree and `f32` cases, and
the grouped shared-descriptor experiment above, do not require a format change.
