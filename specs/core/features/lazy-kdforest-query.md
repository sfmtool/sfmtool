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
for the other randomized trees. Duplication trades disk capacity and write time
for predictable leaf access. It is the main review decision, not a free gain.

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

## Benchmark plan and provisional defaults

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
and record corpus/query hashes and hardware. Publish results before choosing
shipping defaults; retain configurability even if a clear winner emerges.

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

## Out of scope

There are no updates or appends to an existing file, no remote HTTP reads, no
Python bindings, no CLI, no automatic precision calibration, no alternate
metrics, and no external descriptor dependencies. Existing in-memory query
callers keep their current API.

The format and this query path both cover `u8` and `f32`, matching the scalar
types the in-memory forest already supports. The open questions that remain are
measurements, not design: whether T-fold vector duplication is worth its cost
against a shared corpus, and where the chunk-size and cache-budget defaults
land. Both are settled by the benchmark plan above without a format change.
