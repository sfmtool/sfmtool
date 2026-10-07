# Lazy KD-Forest Queries

A `.kdf` file stores a randomized kd-tree forest, an index for approximate
nearest-neighbour search, together with the descriptors it indexes, such as the
SIFT descriptors of a set of images. `LazyKdForest` answers nearest-neighbour
queries against that file directly. It reads and decompresses only the tree
chunks and descriptor blocks a query visits, keeps them in a cache of bounded
size, and returns the same neighbours as the in-memory forest the file was
written from. This lets a process search a descriptor set larger than the memory
it can spend on it, and reuse an index across runs without rebuilding it. The
file stores each descriptor once, however many trees the forest has.

Packing and cache values are configurable. The interface lists their defaults,
[Why the defaults are what they are](#why-the-defaults-are-what-they-are) gives
the reasons, and
[lazy-kdforest-query-measurements.md](lazy-kdforest-query-measurements.md)
records the measurements that chose them.

It extends [randomized kd-tree forests](randomized-kdtree-forest.md)
and uses the [KDF format](../../formats/kdf-file-format.md).

## Rust interface and responsibilities

The lazy forest sits beside the in-memory forest in
[core's kdforest module](../../../crates/sfmtool-core/src/features/kdforest/mod.rs).
The `sfmtool-kdf-format` crate owns storage types and validated decoded chunks;
core depends on it, never the reverse. Persistence exports the already-built
topology and feature order rather than rebuilding from a seed, so the random
generator and the builder are not part of the file compatibility contract.

The public surface is implemented in
[`kdforest/persistent.rs`](../../../crates/sfmtool-core/src/features/kdforest/persistent.rs)
and re-exported by the kdforest module:

```rust
pub struct KdfWriteOptions {
    pub target_descriptor_block_bytes: usize, // default: 2 KiB
    pub target_chunk_bytes: usize,    // default: 1 MiB
    pub compression_level: i32,       // default: 3
    pub origin_block_rows: usize,     // default: 131072 (two u32 columns = 1 MiB)
    pub replace_existing: bool,       // default: false, so a destination that exists is refused
}
pub struct LazyKdForestOptions {
    pub max_address_map_bytes: usize, // default: 768 MiB, 5 bytes a feature
    pub max_leaf_features: usize,     // default: 1,048,576
    pub cache_bytes: usize,           // default: 256 MiB decoded cache
    pub max_in_flight_bytes: usize,   // default: 64 MiB decode reservations
    pub max_compressed_bytes: usize,  // default: 384 MiB, set by the compressed row map
    pub max_metadata_bytes: usize,    // default: 384 MiB
    pub max_chunk_bytes: usize,       // default: 64 MiB decoded
    pub query_workers: usize,         // default: 1; the caller can raise it
}
pub struct LazyKdForest<S: ForestScalar + KdfScalar> { /* file handle, cache, workers */ }
pub type LazyKdForestU8 = LazyKdForest<u8>;
pub type LazyKdForestF32 = LazyKdForest<f32>;

impl<S: KdfScalar> KdForest<S> {
    pub fn write_kdf(&self, path: &Path, sources: Option<&KdfSiftSources>,
                     options: &KdfWriteOptions, progress: &Progress<'_>)
        -> Result<(), KdfError>;
    pub fn write_kdf_ordered(&self, path: &Path, sources: Option<&KdfSiftSources>,
                             options: &KdfWriteOptions,
                             descriptor_order: Option<&[u32]>,
                             progress: &Progress<'_>)
        -> Result<(), KdfError>;
    pub fn read_kdf(path: &Path, options: LazyKdForestOptions)
        -> Result<Self, KdfError>;
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
    pub fn search_batch_with_distances_ordered(&self, queries: &[S],
        n_queries: usize, k: usize, max_leaf_checks: usize,
        max_dist: Option<f32>, order: &[u32])
        -> Result<(Vec<u32>, Vec<f32>), KdfError>;
    pub fn search_batch_with_stats(&self, queries: &[S], n_queries: usize,
        k: usize, max_leaf_checks: usize, max_dist: Option<f32>)
        -> Result<(Vec<u32>, Vec<f32>, LazyQueryStats), KdfError>;
    pub fn self_join_with_distances(&self, k: usize, max_leaf_checks: usize,
        max_dist: Option<f32>) -> Result<(Vec<u32>, Vec<f32>), KdfError>;
    pub fn self_join_with_distances_progress(&self, k: usize,
        max_leaf_checks: usize, max_dist: Option<f32>, progress: &Progress<'_>)
        -> Result<(Vec<u32>, Vec<f32>), KdfError>;
    pub fn new_scratch(&self) -> LazySearchScratch<S>;
    pub fn len(&self) -> usize;
    pub fn dim(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn resolve_origins(&self, feature_ids: &[u32])
        -> Result<Option<Vec<FeatureOrigin>>, KdfError>;
    pub fn resolve_feature_geometry(&self, feature_ids: &[u32])
        -> Result<Option<Vec<FeatureGeometry>>, KdfError>;
    pub fn resolve_descriptors(&self, feature_ids: &[u32]) -> Result<Vec<S>, KdfError>;
    pub fn image_table(&self) -> Result<Option<&KdfImageTable>, KdfError>;
    pub fn io_stats(&self) -> KdfIoStats;
    pub fn reset_io_stats(&self);
    pub fn content_xxh128(&self) -> &str;
}
```

`search_with_stats` returns `LazyQueryStats` — leaf checks, heap pushes and pops
— alongside the neighbors; `search` is the same traversal with the counters
dropped. The counters exist because they are what the parity tests assert on:
matching neighbor IDs alone would not catch a file-backed traversal that visits
a different set of leaves and happens to agree. `io_stats` reports the cache and
read counters of `KdfIoStats` for the whole file, which is how a test asserts
that opening reads no chunk payload and that a warm hit causes no read;
`reset_io_stats` zeroes the cumulative counters so a benchmark can measure one
phase at a time. `content_xxh128` returns the whole-content hash the writer
recorded.

`search_batch_with_distances_ordered` processes the queries in a caller-given
order, which must be a permutation of `0..n_queries`, and still writes each
result to its own row. Only the schedule changes: queries that are close in
descriptor space tend to reach the same chunks and blocks, so the cache serves
them. `search_batch_with_stats` is the batch call with the traversal counters
summed over the batch, which is what read amplification is computed from.
`self_join_with_distances` queries every stored descriptor against the forest,
reading each query from the file rather than taking the corpus as a slice, so a
whole-corpus self-join needs memory for the cache and the `n * k` result table,
not for the corpus; the `_progress` variant reports a count of answered queries
and stops with `KdfError::Cancelled`. `new_scratch` returns a
`LazySearchScratch`, the per-worker queue, dedup sets and buffers that the batch
calls allocate once per worker rather than once per query.

`KdForest::read_kdf` rebuilds a full in-memory forest from the file without
running a build: the file stores the exact topology, leaf membership and feature
IDs, so this is decompression and reassembly. It is the third option beside
querying the file lazily and rebuilding from the `.sift` corpus.

A write of a few million descriptors is several hundred megabytes through zstd,
which is seconds of work, so both carry a `Progress`
([`../../gui/operation-progress.md`](../../gui/operation-progress.md)) down into
the format crate's own `write_kdf`; a caller with nothing to report through
passes `Progress::none()`, which is why there is one entry point per ordering
rather than a reporting sibling beside each. It moves the fraction by where the
write's time goes rather than by how many bytes are behind it, and reads the
cancel flag between batches of blocks; a cancelled write is
`KdfError::Cancelled` and no file, since the archive is streamed into a
temporary sibling and renamed over the destination only once it is whole.

`KdfScalar` is a sealed bridge for the existing u8/f32 scalar implementations,
not an invitation to persist arbitrary user metrics. `Neighbor` retains original
row ID and squared distance. Fallible queries are necessary because an unread
chunk can fail after open succeeds. Errors distinguish I/O (with entry context),
unsupported format/scalar, invalid shape/query/options, resource limit, malformed
structure and integrity mismatch. A chunk or block read checks its decoded size
and the constraints its contents must satisfy; it does not hash them, because a
digest in this format covers a whole section and a query reads a few blocks out
of millions. Hashing is what the offline verifier below does. A failed query or batch returns an error, not
an apparently successful shortened result.

Illustrative usage, with a file built from the same in-memory index:

```rust
use sfmtool_core::features::kdforest::{
    KdForestU8, KdForestParams, KdfWriteOptions,
    LazyKdForestU8, LazyKdForestOptions,
};
use sfmtool_core::progress::Progress;
let features = vec![0u8, 0, 10, 10, 1, 1];
let forest = KdForestU8::build(&features, 3, 2, KdForestParams::balanced(), &Progress::none())?;
let options = KdfWriteOptions::default();
forest.write_kdf("example.kdf".as_ref(), None, &options, &Progress::none())?;
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
pub fn kdf_summary(path: &Path, max_metadata_bytes: usize)
    -> Result<KdfSummary, KdfError>;
```

`kdf_summary` accounts for a file's size by section from the ZIP central
directory and the decoded metadata alone, without decoding any payload. It is
not generic over the scalar type, so it can describe a file whose scalar type
the caller does not yet know.

`verify_kdf` checks the self-contained archive. `verify_sift_sources` is the
explicit, separate audit against the live workspace recorded in its provenance.
Writing uses a sibling temporary file and publishes only a completed archive; it
fails if the destination exists, unless `KdfWriteOptions::replace_existing` says
it may be replaced, in which case the same rename puts the new archive over the
old one and the old one stands until that instant. Replacing is how an index is
rebuilt in place over the file a reader still holds open
([`../../gui/sift-index.md`](../../gui/sift-index.md)).
Streaming construction from an out-of-memory
input corpus is out of scope: export requires an already-built forest.

### Source references

`KdfSiftSources` contains the format's workspace configuration, image names and
both per-image hashes, parallel image/image-feature index arrays, and N rows of
keypoint/affine geometry in original input order. Writers validate complete
coverage, finite geometry, and pair uniqueness.
`FeatureOrigin` contains `image_index: u32` and `image_feature_index: u32`; an image
table accessor exposes the names and hashes without opening source files.
`resolve_origins` returns mappings in requested ID order, including repeated
requests, returns `None` for a generic corpus, and rejects out-of-range IDs.
`resolve_feature_geometry` follows the same contract and returns rows shaped as
`[[x, y], [a11, a12], [a21, a22]]`.

`resolve_descriptors` is the same contract for the descriptor corpus itself, so a
caller holding feature IDs rather than descriptors reads them back here instead
of reopening a `.sift` file; consecutive IDs in one block share a cache pin. The
[constellation query](kdf-constellation-query.md) is what wants all three, and
what the eager forest's matching `resolve_descriptors` exists for.

Origin and geometry blocks are cached on demand under the same byte budget as tree chunks;
only blocks covering requested result IDs are read. Image metadata/hashes load
on first source-table access, within the metadata budget. Normal ANN reads
neither origins, geometry, nor image tables. Offline KDF verification recomputes
every section digest from the bytes on disk and checks origin ranges, uniqueness,
and every geometry block; a separate explicit source verifier checks referenced
SIFT hashes, feature bounds, vector equality, and keypoint/affine equality,
reporting missing files distinctly.
Repacking preserves this mapping. Exporting a descriptor subset retains its
original SIFT feature indices even though input row IDs are newly dense.

## One corpus, one search implementation

The format, at version 3, has one storage layout. `KdfWriteOptions::default()`
selects the descriptor, tree-chunk, origin, and compression defaults; callers may
tune block sizes but cannot duplicate vectors per tree. Files of any other
version are rejected.

There is one best-bin-first traversal, result set, dedup set and scalar distance
implementation. Storage access supplies node data, ordered leaf feature IDs and
vector rows. A feature ID indexes the resident storage-row map, which locates a
descriptor block. This keeps storage addressing out of the ANN algorithm.

Export orders descriptors by tree 0's leaf permutation, writes its inverse
as storage_rows, and partitions vectors into Q complete rows per block, where
`Q = max(1, floor(target_descriptor_block_bytes / bytes_per_vector))`. A zero
target is rejected. Targets smaller than a row produce one-row blocks. This block target is
independent of the tree-chunk target and does not change topology or leaf size.

Open loads and validates the row map within `max_address_map_bytes`,
checking permutation validity with bounded temporary memory. A map over that
limit is rejected rather than paged. The resident map and validation-scratch
bytes are reported; the map is separate from the decoded chunk cache and metadata limit.

Descriptor, optional geometry, tree, and origin blocks use the same byte-weighted
cache with distinct key kinds and single-load coordination.
Before requesting descriptors, a query copies the ordered leaf IDs into its
scratch and releases the tree chunk pin. It processes rows in leaf order,
deduplicating IDs before fetching vectors, and releases a descriptor block pin
before requesting another block, so it never waits for cache admission while
holding a different cache pin. Leaf scratch is additional memory bounded by
`max_leaf_features`; a larger leaf is rejected. The order in which blocks are
read never changes the order of evaluations or result ties.

The geometry corpus uses the same storage permutation and row boundaries as
descriptors, so block b in each corpus correlates positionally. Its cache key,
compressed frame offsets, and hashes are independent. A constellation consumer
can therefore fetch the geometry for result IDs without opening `.sift` files.

## Packing policy

The format permits arbitrary chunk boundaries. The exporter computes subtree
packing weights from the node, split, and feature-ID arrays actually stored in
tree chunks. Descriptor and geometry blocks use their separate row target.
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
Chunk IDs are assigned deterministically, root chunk first, then the remaining
chunks in left-before-right traversal order. Physical ZIP order follows
tree/chunk IDs. The writer never changes leaf size to make chunks fit, because
that would change the ANN index.

For a binary median tree, maximal fitting subtrees often underfill the target,
so a chunk is described by its actual size, not by the 1 MiB target. Packing is
by decoded size, never compressed size: compression depends on the data and
would make allocation sizes and benchmark comparisons misleading.

The corpus defaults to tree-0 leaf order. Other trees may scatter one leaf across
several descriptor blocks; the measured cost is outweighed by removing `T-1`
descriptor copies. Callers may provide another explicit corpus permutation
without changing feature identity or query results.

## Search behavior and parity

The behavioural reference is the in-memory search in
[search.rs](../../../crates/sfmtool-core/src/features/kdforest/search.rs), not the
older pseudocode's equality convention. The storage path preserves the following
details:

- Every tree is seeded in increasing tree index before the check budget is
  tested. A descent goes left for query coordinate <= split, right for > split.
  For float32 it uses the existing scalar total-order comparison, including
  signed-zero behavior.
- A descent retains its incoming branch priority along its near chain; far
  children are enqueued with incoming priority plus squared split-plane
  distance, only when this is <= the current result threshold.
- The smallest priority pops first. Equal priorities use descending tree index,
  then descending **logical node ID**, matching the existing heap tuple order.
  Chunk IDs never replace logical node IDs in this comparison.
- Original feature IDs are deduplicated across trees and leaf members are
  evaluated in their stored order. Check count measures unique distance
  evaluations. A leaf, once entered, is finished even when it crosses the
  budget; zero checks still seeds trees.
- After seeding, the search stops before the next descent if checks meet the
  budget or its priority exceeds the current threshold. The additive priority is
  not an admissible geometric bound: unlimited checks still do not imply exact
  ANN.
- Result ties behave as in memory: equal distances retain encounter order, and
  an equal candidate does not replace the worst member of a full result set. The
  scalar kernels and cutoff conversion are the in-memory implementation's.

Cache hits, eviction, file layout, and worker schedule affect speed only.
For the same persisted topology and supported finite inputs, results and check
counts match the in-memory reference. Float results are bit-identical when
using the same scalar kernel/platform; no cross-platform float promise is added.
NaN/infinite query coordinates and negative/NaN cutoffs are rejected; a positive
infinite cutoff means unbounded. Finite float coordinates may overflow squared distance to
positive infinity, as in the existing kernel. Empty forests and k = 0 return
empty results without reading tree chunks. A wrong query dimension is an error.
Batch arrays are row-major with `u32::MAX`/positive infinity padding.

There is no independent I/O cutoff that silently truncates ANN. A future
latency-budgeted search would need to report incompleteness explicitly.

## Reads, cache, and memory accounting

Open reads the ZIP central directory, metadata and content-hash entry once,
builds a chunk-to-entry index, and validates their bounded sizes. It does not
decode trees or descriptors. ZIP metadata/index memory is O(number of entries),
not constant; the metadata budget applies to both decoded JSON and index
allocations, and excess directory entries are rejected before unbounded
allocation.

A tree miss reads and checks only that chunk's grouped topology/feature-ID
entry. Descriptor and geometry misses read one independent frame from their
respective corpora; an origin miss reads its two compressed columns.
A chunk is a logical cache unit, not necessarily one system call. The
offset/length index is kept for the handle's lifetime; the ZIP is never reopened
or reparsed for a node. Descriptor and geometry frames are read with positional
reads, so concurrent misses share no seek cursor, and are decompressed outside
any lock; tree chunks and origin blocks are read and decompressed through one
ZIP reader behind a lock. The eager `DecodedEntries` path,
which loads the whole archive, is not used.

The cache is a per-file byte-weighted LRU of decoded chunks and blocks, keyed by
payload kind and block/chunk ID. In-flight requests for the same key share one
load. No "upper tree" is pinned: frequent routing chunks stay in the cache
because they are reused.
Each descent holds at most its current decoded chunk and releases it before
requesting another. Queue entries hold addresses, not chunk references.

The cache budget includes pinned resident chunk arrays; admission reserves space
and waits for readers to release chunks if necessary. The cache and in-flight
limits must each admit the largest declared chunk, and a declared chunk over
`max_chunk_bytes` is rejected. The same limits apply to descriptor, geometry,
and origin blocks. Decode memory is reserved before I/O. Each compressed buffer is
bounded separately by `max_compressed_bytes`; these buffers and the decoder workspace are
outside the decoded cache limit, which is therefore not a process RSS limit. A loader never
waits for admission while holding a different chunk pin, avoiding cache deadlock.

`cache_bytes` needs to cover the query's working set, not just the largest item
the file declares. The 256 MiB default admits the blocks of a 9.7M-descriptor
index, but repeated patch searches against such an index evict blocks they need
again
([measurements](lazy-kdforest-query-measurements.md#cache-budget-for-repeated-searches)).
Callers with a large index should compare repeated-query latency at several
budgets before attributing that latency to forest search itself.

Per-query dedup uses reusable bitsets for feature IDs and logical node IDs, with
lists of touched words to clear between queries. Scratch includes bits proportional
to corpus and tree size per Rayon job. Queue memory grows with explored branches. Batch output is O(M*k);
metadata, scratch, output, compressed buffers and decoder workspace are additional
to the cache. Checked arithmetic and configurable worker count constrain growth;
this is bounded residency of file data, not constant total memory for arbitrary k.

An open file is treated as immutable for the handle's lifetime. Atomic replacement can
leave an old handle serving its old snapshot; in-place writes are unsupported.
Chunk validation checks references before dereference and detects revisited
logical nodes per query to prevent malformed cycles. Full semantic verification
is an explicit offline operation, never implicit at lazy open.

## Where this sits in the literature

[Beis & Lowe 1997](https://www.cs.ubc.ca/~lowe/papers/cvpr97.pdf) describes
Best-Bin-First search with a bounded search effort.
[Lowe 2004](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf), § 7.2, applies it to
SIFT matching with a cutoff of 200 candidate checks.
[Muja & Lowe 2009](https://www.cs.ubc.ca/~lowe/papers/09muja.pdf) describes
randomized kd-trees searched through a shared priority queue. These are the
algorithmic references for this implementation; they do not evaluate its
compressed, file-backed access path.

[Muja & Lowe 2014](https://www.cs.ubc.ca/~lowe/papers/14mujaPAMI.pdf), § 5,
considers disk-backed data among the options for corpora larger than memory and
implements distributed nearest-neighbor search across machines. The
[measurements](lazy-kdforest-query-measurements.md) for this path evaluate a
different configuration: one machine, a local file, and a bounded decoded cache. They do not compare against distributed FLANN or establish how
storage hardware changes that comparison.

The recall metrics also differ. Lowe reports loss of correct matches after the
ratio test on a 100,000-keypoint database; the measurements for this path
include raw recall@1 on larger corpora. Raw recall alone does not establish how many usable
matches this pipeline loses. That requires measuring the downstream matcher.

## Access path

[`persistent.rs`](../../../crates/sfmtool-core/src/features/kdforest/persistent.rs)
borrows one tree chunk while following nodes within it. At a leaf it copies
unchecked feature IDs into reused scratch, releases the tree pin, and borrows
consecutive descriptors from each corpus block. A query holds no pin while
admitting another block. Descriptor encounter order, queue priorities and check
counts are unchanged, including equal-distance ties.

[`cache.rs`](../../../crates/sfmtool-kdf-format/src/cache.rs) maintains an indexed
doubly linked LRU list per shard. Hits and victim removal take constant time;
eviction walks only older pinned entries before an available victim. Recency slots
are reused. Pins borrow the cache owner rather than modifying a global reference
count; an `Arc` still pins each value and its bytes count against residency. A
miss-only global gate enforces the total in-flight decode limit without reducing
cache sharding. Pin-release notifications synchronize with the admission mutex,
and waiters register before checking capacity.

The cache is split into up to 16 independently locked shards, because every node
and descriptor a query touches takes a cache lock, and with one lock query workers
only wait on each other. The shard count is not free to choose. Admission is per
shard, so a shard smaller than the largest item a caller may ask for could never
admit it and the caller would block forever; the count is therefore capped at
`cache_bytes` divided by the largest validated item in the file, not by the
caller's permissive `max_chunk_bytes` ceiling. A cache only just large enough for
that item collapses to a single shard. Keys are dense block and chunk indexes, so
their low bits spread uniformly across shards with no hashing, and tree IDs
contribute low bits so that different trees' root chunks use different locks.
Sharding on the cached contents instead, such as a descriptor's leading bytes,
would be badly skewed, because SIFT descriptors carry many small and zero
components. The maps are keyed with XXH3 rather than the standard library's
SipHash-1-3: SipHash resists hash flooding from attacker-chosen keys, these keys
are dense integers this crate generates while walking a file it has already
validated, and the crate already hashes every stored section with XXH3.

The descriptor and geometry corpora use explicit-offset reads: `read_at` on Unix
and `seek_read` on Windows, with retries for interrupted or short reads and an
error for premature EOF. The handle's cursor is never used to choose a frame, so
no lock is held around a seek. Each executing thread reuses a zstd decompressor
context; its workspace is additional to decoded-cache residency and lasts with
that thread. On Windows, positioned reads on a single synchronous handle still
serialize, so the corpus opens a bounded pool of independent handles, sized by
`query_workers`, and assigns executing threads to stable slots. It uses
[ReOpenFile](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-reopenfile)
on the existing handle, which keeps the original file-system object if the path
is atomically replaced; duplicating a handle with `try_clone` would share
synchronous I/O state. Unix uses `read_at` on the original snapshot without a
handle pool.

Eager reload validates the reconstructed graph for cycles, duplicate or missing
features and unreachable nodes before handing it to the unchecked in-memory
traversal. Provenance includes the default check budget, which reload restores;
older files without this field retain the balanced fallback.

Batch queries write directly into preallocated result rows. Ordered batches and
self-joins restore row order by cycling the permutation in place, avoiding a
second `N * k` result table and one neighbor-vector allocation per query. The
permutation requires O(N) IDs.

The lazy path costs more than the in-memory forest in two separate ways. With a
resident working set, it validates addresses, tracks visited nodes, resolves
storage rows and acquires pins, where the in-memory traversal indexes resident
arrays directly; disk speed cannot explain a run with zero reads. Under cache
pressure, misses also need admission, reads and decompression, and eviction can
make the same block decode repeatedly.

## Why the defaults are what they are

Each default below was chosen by a measurement in
[lazy-kdforest-query-measurements.md](lazy-kdforest-query-measurements.md),
which says why that measurement was taken and what it decided.

- **One shared descriptor corpus.** It is 3.3-3.4x smaller than a copy per tree
  at every scale measured, faster or equal in every regime measured, and at full
  speed on a cache a third the size. A copy per tree was faster only for a fully
  resident warm batch, and a caller in that position can reload the file into the
  in-memory forest
  ([measurements](lazy-kdforest-query-measurements.md#shared-corpus-versus-per-tree-copies)).
- **Tree-0 leaf order for the corpus.** Two alternative orders, a Z-order curve
  over principal components and greedy co-occurrence packing, did not read fewer
  blocks over a batch on the largest corpus
  ([measurements](lazy-kdforest-query-measurements.md#corpus-order)).
- **2 KiB descriptor blocks.** Reads from trees other than tree 0 are scattered,
  so a smaller block decodes less unwanted data per descriptor. 2 KiB with
  16-feature leaves gave the fastest whole-image queries on DinoLedge under a
  256 MiB cache, at 2.8% more file than the 4-8 KiB range
  ([measurements](lazy-kdforest-query-measurements.md#descriptor-block-size)).
- **A 1 MiB tree-chunk target.** Tree chunks carry no descriptors, so chunk size
  has little effect on cold or sparse queries
  ([measurements](lazy-kdforest-query-measurements.md#tree-chunk-size)).
- **16-feature leaves.** At equal recall, leaf 16 is fastest when the cache holds
  a small part of a large index; leaves of 32 to 64 are faster when the cache
  holds nearly the whole stored working set
  ([measurements](lazy-kdforest-query-measurements.md#leaf-size)).
- **A 256 MiB decoded cache.** It admits the blocks of the largest index
  measured, but it is not a working-set size: repeated searches against a
  1.4 GB index ran 37x faster on a second identical search at 512 MiB
  ([measurements](lazy-kdforest-query-measurements.md#cache-budget-for-repeated-searches)).

## Which workloads this path suits

The file-backed path suits sparse queries. A patch localization reads thousands
of blocks, not hundreds of thousands. Its query costs 20-40x the in-memory
query, but that is a few percent of the whole job, which geometric verification
dominates. Opening a
`.kdf` is two orders of magnitude faster than building or loading a forest, so a
caller with a few queries is ready almost at once.

It does not suit a workload that visits most of the corpus. A whole image of
8,192 descriptors reaches about 92% of a large file's descriptor blocks, so the
cache must hold nearly the whole file or it evicts and re-decodes continuously.
An algorithm that queries every descriptor pays a lock, a hash lookup and an
`Arc` clone per descriptor, about 47 us per query against about 1 us in memory,
and no cache tuning removes that. Such a caller should load the forest with
`KdForest::read_kdf` or give the cache the whole file. Loading is itself slower
than rebuilding from `.sift`, because it scatters every descriptor into
feature-ID order. The measurements are in
[Which workloads suit the file-backed path](lazy-kdforest-query-measurements.md#which-workloads-suit-the-file-backed-path).

## Testing

Parity is the property that carries the risk here, so the tests assert it
against the in-memory forest rather than against recorded expectations. Each
case builds a forest, exports it, opens the file, and compares. Chunk and block
targets are set to a few hundred bytes so that a forest of a few dozen
descriptors still spans many chunks: the traversal that matters is the one that
crosses a chunk boundary, and a realistic target would put the whole fixture in
one chunk and test nothing.

[`persistent/tests.rs`](../../../crates/sfmtool-core/src/features/kdforest/persistent/tests.rs)
holds six cases. The `u8` case exports a four-tree forest and
compares eleven queries at leaf budgets of 0, 1, 7, 31 and 1000, plus a batch
call. It asserts equal leaf-check counts as well as equal neighbors, because
matching neighbor IDs alone would not catch a file-backed traversal that visits
a different set of leaves and agrees by luck — the counts are what pin the
far-child queue order, and with it the logical-ID tie-break. The `f32` case
covers signed-zero routing at a split plane, an infinite cutoff, and the two
rejected queries (NaN coordinate, negative `max_dist`). The concurrency case
gives eight threads one identical query against a cache smaller than the file,
so every load races, evicts and is deduplicated; it asserts that opening a file
reads no descriptor block at all. The reload case writes a forest, reads it back
with `KdForest::read_kdf`, and checks that the reloaded forest answers queries
identically. The ordered-read case builds a forest of identical descriptors in a
reversed storage order, checks that `search_batch_with_distances_ordered` returns
the same ties as the in-memory batch, and checks that it rejects a schedule with
a repeated or out-of-range query index. The last case checks that rebuilding a
tree from the file rejects a node cycle and a feature that no leaf or more than
one leaf covers.

[`sfmtool-kdf-format/src/tests.rs`](../../../crates/sfmtool-kdf-format/src/tests.rs)
covers the format in isolation: a round trip down to node
addresses and logical IDs, a full `verify_kdf` pass, SIFT origins resolved
lazily and in the caller's requested order including repeats, the writer's
refusal to overwrite an existing destination, and a `ResourceLimit` when the
address-map budget cannot hold the row map. The corruption case is the
one that pins the laziness contract: a damaged descriptor block leaves
`open` succeeding and reading nothing, makes the direct access fail on the frame
it cannot decode, and makes full verification fail on the corpus section digest.

[`sfmtool-kdf-format/src/validation_tests.rs`](../../../crates/sfmtool-kdf-format/src/validation_tests.rs)
builds the negative surface around a hash-aware archive mutator. It changes a
decoded entry, recomputes the affected section and whole-file digests independently
of the writer, and rewrites the archive. That makes malformed child references,
cycles and shared children, bad leaf ranges, reserved fields, invalid tree and
storage permutations, split violations, nonfinite descriptor and geometry
`f32` values, unsupported metadata, wrong entry sets, duplicate ZIP names and
truncated frames reach the validator each test names instead of stopping at an
unrelated integrity mismatch. It also asserts that an unvisited chunk stays unread
and a warm resident access performs no read or decode.

The Python binding test extracts SIFT once from the included 270x480 Seoul Bull
image and reuses that file for every `verify_sift_sources` case: matching and
relocated workspaces, a missing source, a changed identity, an out-of-range source
feature, a mismatched descriptor, and mismatched keypoint geometry. The same missing-source case confirms ordinary
queries and embedded origin lookup remain available. Synthetic origin mutation in
the Rust suite covers out-of-range image IDs and duplicate source pairs without
another extraction.

## The Python surface

The [measurements](lazy-kdforest-query-measurements.md) are Python jobs —
sweeps over corpora, block sizes, chunk sizes and cache budgets, reporting
latency percentiles and recall — so the
`uint8` path is bound on the `sfmtool.spatial` submodule, in
[`spatial/kdf.rs`](../../../crates/sfmtool-py/src/spatial/kdf.rs), beside the
in-memory `KdForest` it is measured against. Only `uint8` is bound: the Python
`LazyKdForest` wraps `LazyKdForestU8`, and opening a `float32` file with it
raises `ValueError`, because the file's scalar type does not match.

```python
from sfmtool._sfmtool.spatial import (
    KdForest, LazyKdForest, write_kdf, kdf_file_summary,
    verify_kdf, verify_sift_sources,
)

forest = KdForest(descriptors, num_trees=4, leaf_size=16, seed=7)
write_kdf(forest, "corpus.kdf", sources=sources)  # default: 2 KiB blocks

lazy = LazyKdForest("corpus.kdf", cache_bytes=256 << 20, query_workers=4)
indices, distances, stats = lazy.query_with_stats(queries, k=2, max_leaf_checks=128)
io = lazy.io_stats()
amplification = io["decoded_bytes"] / (stats["checks"] * lazy.dim)
geometry = lazy.resolve_feature_geometry(indices.ravel())
```

Three things about that surface follow from what it is for rather than from
the Rust API it wraps.

There is no `layout` argument. `descriptor_block_bytes` tunes the
only corpus representation, and SIFT `sources` includes `(N,2)` positions plus
`(N,2,2)` affine shapes.

`reset_io_stats` exists because the alternative for separating an open from
the queries after it, or a cold pass from a warm one, is reopening the file,
and reopening also drops the cache. The counters zero; what the cache holds
does not.

`kdf_file_summary` splits a file per role rather than reporting one total,
reading only the ZIP central directory and the metadata entry, so it costs the
same on a 5 GB file as on a 5 KB one. It attributes `descriptors`,
`feature_geometry`, their offset tables, the `storage_row_map`, `origins`, and
`tree_chunks` separately.

Errors are split by what a sweep must do about them: a budget that cannot hold
what was asked for raises `MemoryError` (try another cell), a malformed or
damaged file raises `OSError` (stop), and a bad argument or a file of the
wrong scalar type raises `ValueError`.

## Out of scope

There are no updates or appends to an existing file, no remote HTTP reads, no
CLI, no automatic precision calibration, no alternate metrics, and no external
descriptor dependencies. Existing in-memory query callers keep their current
API. The Python bindings cover `uint8` only.

The format and this query path both cover `u8` and `f32`, matching the scalar
types the in-memory forest already supports.

## Open questions

- Whether the four-tree defaults hold at twenty trees, and for `f32`
  descriptors, is unmeasured. Neither case would need a format change; see
  [Not yet measured](lazy-kdforest-query-measurements.md#not-yet-measured).
- Grouping nonconsecutive descriptor requests by block is an experiment in
  [drafts/kdf-shared-descriptor-reads.md](../../drafts/kdf-shared-descriptor-reads.md).
  It must preserve encounter-order ties.
