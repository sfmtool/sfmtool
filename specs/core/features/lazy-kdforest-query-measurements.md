# Lazy KD-Forest Query Measurements

This file records the measurements behind the design and defaults of the
[lazy KD-forest query path](lazy-kdforest-query.md). That path answers
nearest-neighbour queries straight from a `.kdf` index file, reading only the
parts a query visits into a bounded cache. The measurements decided four kinds
of things:
- the storage layout: one shared descriptor corpus instead of a copy per tree;
- the writer's block and chunk sizes and the builder's leaf size;
- how the cache and the file reads are built;
- which workloads the file-backed path suits at all.

The spec states each decision and links here. This file holds the evidence, and
for each measurement says why it was taken and what its result decided. The
file-size measurements of the same layouts are in
[kdf-layout-measurements.md](kdf-layout-measurements.md).

| Decision | Outcome | Section |
|---|---|---|
| Per-tree descriptor copies or one shared corpus | One shared corpus, the only layout from format version 2 | [Shared corpus versus per-tree copies](#shared-corpus-versus-per-tree-copies) |
| Order of the stored corpus | Tree-0 leaf order | [Corpus order](#corpus-order) |
| Descriptor block size | 2 KiB | [Descriptor block size](#descriptor-block-size) |
| Tree chunk size | 1 MiB target | [Tree chunk size](#tree-chunk-size) |
| Leaf size | 16 features | [Leaf size](#leaf-size) |
| Cache recency structure | Indexed doubly linked LRU list per shard | [Cache hit cost](#cache-hit-cost) |
| Cache sharding | Up to 16 shards | [Query workers and cache sharding](#query-workers-and-cache-sharding) |
| How descriptor blocks are read | Positioned reads; a handle pool on Windows; a reused zstd context per thread | [Reading descriptor blocks](#reading-descriptor-blocks) |
| Default decoded cache size | 256 MiB, with guidance to measure larger budgets | [Cache budget for repeated searches](#cache-budget-for-repeated-searches) |
| Where to use a `.kdf` rather than an in-memory forest | Sparse queries such as patch localization; not whole-corpus algorithms | [Which workloads suit the file-backed path](#which-workloads-suit-the-file-backed-path) |

## How the measurements are taken

**Why this method.** A storage choice should change only speed and size, never
the answers. So each comparison holds the forest, the queries, `k` and the check
budget fixed and varies only storage. Every run also checks that the file-backed
results equal the in-memory forest's. A measurement that changed the answers
would be measuring a different index.

One MiB is 1,048,576 decoded bytes. Packing is compared by decoded size. A
comparison holds vectors, topology, query order, k and check budget fixed. The
axes the sweeps draw from:

| Axis | Cases |
|------|-------|
| Corpus | Real 128-D SIFT at 100k, 1M, and a size exceeding RAM; clustered/duplicate byte data; finite 32-D and 128-D float data |
| Forest | 4, 8 and 20 trees; leaf sizes 8, 16, 32; checks 32, 128, 512; k = 1, 2, 10 |
| Query | Independent held-out queries; self-query with self excluded for recall; random and locality-grouped batches; single-query and large batches |
| Cache | 64, 256, 1024 MiB where limits admit chunks; cache smaller than total corpus; worker counts 1, 4, 8 |
| Storage | Local SSD; HDD if available; record OS, device, filesystem and compression level |
| Warmth | Fresh process/application cache with warm OS cache; controlled cold OS cache where available; fully warm repeat |

The sweep is staged rather than the entire Cartesian product. Chunk sizes are
screened first on four trees and 16-feature leaves, and then the best candidates
are stressed. A reopened file is not evidence of cold physical storage, so a
cache state that was not controlled is labelled as such. "Cold" in this file
means a fresh reader with an empty application cache and nothing stronger; the
OS page cache was never flushed.

The quantities recorded are:
- open time and resident metadata;
- p50/p95/p99 query latency and batch throughput;
- compressed bytes requested, decoded bytes and read calls;
- unique chunk misses, cache hits and evictions, and duplicate-load suppression;
- peak cache, pinned, in-flight, scratch and RSS bytes;
- file size and export time.

Read amplification is decoded bytes divided by unique evaluated vector bytes.
It is measured on a **single** cold query. Within one query the check set is
deduplicated, so `checks x dim` is exactly the unique evaluated bytes. Across a
batch it is not, because later queries re-evaluate descriptors that are already
decoded, so batch results report decoded bytes and read counts directly.

Recall is measured against exhaustive search on a held-out subset.
[`scripts/benchmark_kdf_layouts.py`](../../../scripts/benchmark_kdf_layouts.py)
runs the staged sweep against a workspace's `.sift` files. It builds the forest
once and exports it repeatedly, so every cell compares storage against an
identical forest.

**Timings drift, so comparisons are interleaved.** Timings on the measurement
machine drift by up to 2x between runs. During this work, one sample suggested
that a change was worth 3.2x. Repeated runs put the same configuration at half
that speed, and a third run put it back. Comparisons are therefore
*interleaved*: every arm is measured in every round, so drift is shared rather
than attributed to one arm. They are reported as medians with ranges. Treat a
timing claim that was not measured this way with suspicion.

**Results do not depend on storage.** Every cell returned neighbours and
distances identical to the in-memory forest. Recall@1 against exhaustive search
was the same across layouts within a corpus: 0.433, 0.557 and 0.657 at a
128-leaf budget on the three corpora below.

## Shared corpus versus per-tree copies

**Question.** Should each tree store its own copy of the descriptors (the
version-1 tree-local layout) or should all trees share one corpus behind a
feature-ID-to-row map?

**Why these measurements.** The shared layout saves space by construction, but
a cost model made before it was built predicted it would cost cold-query I/O
(see [The pre-build cost model](#the-pre-build-cost-model-2026-09-09)). Only a
measurement on real corpora could say which effect dominates. Three corpora
spanning 276x in size test whether the result depends on scale. DinoLedge, at
9.7M descriptors, is the largest real capture available and is larger than the
cache budgets tried.

**Code state.** Format version 1, which supported both layouts, before the
2026-09-11 access-path revision. The file then held one digest per block; the
format now holds one per section.

Four trees, 16-feature leaves, k = 2, a 128-leaf check budget and zstd level 3.
Timings are medians of repeated runs on one Windows desktop with a local NVMe
SSD.

| Corpus | Images | Descriptors | Tree-local | Shared | Ratio |
|--------|--------|-------------|-----------|--------|-------|
| `seoul_bull_sculpture` | 17 | 35,167 | 13.94 MiB | 4.08 MiB | 3.42x |
| `dino_dog_toy` | 85 | 694,320 | 262.18 MiB | 78.07 MiB | 3.36x |
| DinoLedge | 1,196 | 9,701,948 | 3,840.61 MiB | 1,157.26 MiB | 3.32x |

The shared corpus is 3.3-3.4x smaller at every scale. It is less than 4x
because the tree node, split and feature-ID arrays are stored either way, and
the shared layout adds a row map: 32.8 MB compressed at 9.7M features, against
2.8 GB saved. Descriptors compress to 76.4% in kd-tree leaf order, just below
the 76.96-76.98% of image order, so the image-order estimate used in
[kdf-layout-measurements.md](kdf-layout-measurements.md) was sound.

The first result was a cache defect, not a layout result. A cache hit cost time
proportional to the number of resident entries (see
[Cache hit cost](#cache-hit-cost)), and that charged the shared layout hardest,
because it holds many small blocks and looks up each descriptor separately. The
numbers below are after that fix. The layout comparison reversed when the fix
landed.

Cold and warm times for a 1,000-query batch against DinoLedge with a 4 GiB
budget, and a 2,000-query batch against `dino_dog_toy` with a 256 MiB budget:

| Chunk target | DinoLedge tree-local | DinoLedge shared | `dino` tree-local | `dino` shared |
|---|---|---|---|---|
| 256 KiB | 3.23 s / 0.04 s | 2.56 s / 0.08 s | 1.36 s / 1.14 s | 0.30 s / 0.10 s |
| 1 MiB | 4.37 s / 0.04 s | 2.44 s / 0.08 s | 3.17 s / 2.99 s | 0.27 s / 0.10 s |
| 4 MiB | 10.01 s / 6.38 s | 5.17 s / 0.08 s | 14.95 s / 15.05 s | 0.27 s / 0.10 s |
| 8 MiB | 14.97 s / 11.43 s | 3.63 s / 0.08 s | 29.11 s / 29.63 s | 0.24 s / 0.10 s |
| 16 MiB | 25.19 s / 21.59 s | 3.63 s / 0.07 s | 57.99 s / 58.48 s | 0.25 s / 0.10 s |

The shared columns barely move. The tree-local columns vary 43x on
`dino_dog_toy` across the same chunk range, because four descriptor copies do
not fit the budget and larger chunks make each eviction cost more to undo.
Tree-local wins only one cell: a fully resident warm batch, 0.04 s against
0.08 s on DinoLedge, where it scans a leaf's descriptors in place while the
shared path looks them up one at a time.

The shared layout also reaches full speed on a smaller budget. On
`dino_dog_toy` it is at full speed by 256 MiB (0.24 s), while tree-local needs
1 GiB (0.48 s) and takes 8.03 s at 64 MiB. A file 3.3x smaller fits a cache
3.3x sooner.

**Decision.** Format version 2 kept only the shared corpus, and version 3 keeps
it. It was 3.3x smaller, faster or equal in every regime measured, insensitive
to a chunk-size choice that changed tree-local's time by 43x, and at full speed
on a budget a third the size. Tree-local's resident warm case was faster, but
keeping a second wire representation for that one case would add permanent
reader, verifier and test complexity. A caller whose working set fits in memory
can reload the file into the in-memory forest instead.

### The pre-build cost model (2026-09-09)

**Why it is kept.** Before the shared layout existed, this model estimated what
it would cost. It is the reason the block size was measured separately from the
tree-chunk size, and its prediction is a useful contrast with what the
measurements found.

The model used the DinoLedge inventory in the
[layout study](kdf-layout-measurements.md#version-1-dinoledge-layout-study-2026-09-09)
(9,702,948 descriptors, four trees, 16-feature leaves). It held the trees,
traversal and check budget fixed, so recall could not change. Calculated
tree-chunk counts for that forest:

| Target decoded size | Subtree chunks per tree | Features per subtree | Actual subtree size | All ZIP entries, including source mapping |
|---------------------|-------------------------|--------------------|---------------------|-------------------------------------------|
| 1 MiB | 2,048 | 4,737–4,738 | 0.629 MiB | 32,940 |
| 4 MiB | 512 | 18,951–18,952 | 2.515 MiB | 8,364 |
| 8 MiB | 256 | 37,902–37,903 | 5.029 MiB | 4,268 |
| 16 MiB | 128 | 75,804–75,805 | 10.058 MiB | 2,220 |

Each tree also has one routing chunk. The routing working set across four trees
is 270,204 bytes at the 1 MiB target and 16,764 at 16 MiB, so the root chunks
can stay in the cache without pinning whole trees. Seeding a cold four-tree
query touches one subtree per tree plus routing: about 2.77 MB decoded at the
1 MiB target against 42.2 MB at 16 MiB.

Storage estimates, in decimal GB:

| Storage estimate | 4 trees | 20 trees |
|---|---|---|
| Duplicated decoded trees/vectors | 5.400 | 27.000 |
| Shared decoded trees/vectors plus address map | 1.713 | 3.441 |
| Decoded reduction | 68% | 87% |
| Compressed descriptor copies removed, at proxy ratio 0.772 | 2.876 | 18.217 |

For cold queries, the model assumed every descriptor outside tree 0 falls in a
uniformly random one of B = 2,048 blocks. With q = (T - 1) x 9.253 descriptors,
the expected number of distinct blocks is `1 + (B - 1)*(1 - (1 - 1/B)^q)`:

| Seeding only, 1 MiB target | 4 trees | 20 trees |
|--------------------------|---------|----------|
| Leaf member visits, before dedup | about 37 | about 185 |
| Duplicated subtree blocks fetched | 4 | 20 |
| Shared descriptor blocks, scattered model | about 29 | about 169 |
| Duplicated decoded bytes, including routing | 2.77 MiB | 13.86 MiB |
| Shared decoded descriptor bytes, excluding tree reads | 16.52 MiB | 97.99 MiB |

With subtree-sized descriptor blocks, the model predicted about 6-8x more cold
transfer and decode work for the shared layout. It concluded that shared
descriptors should get their own, much smaller block size, independent of the
tree chunks, and be measured before the T-fold duplication was accepted.

**What the measurement found instead.** With small, separate descriptor blocks
the shared layout was faster cold as well as warm. In the model, a large block
cost a lot to decode for each scattered descriptor, and small blocks removed
that cost. The prediction was right about scatter (see
[Corpus order](#corpus-order)) and wrong about its price once blocks were small.

## Corpus order

**Question.** In what order should the writer store the shared corpus? The
row map is an explicit permutation, so this is a writer policy with no format
consequence.

**Why these measurements.** The shared corpus is stored in tree-0 leaf order.
Tree 0's leaves are therefore contiguous, but every other tree partitions the
corpus differently. Counting blocks read per check shows how much each extra
tree scatters reads. Two other orderings were then measured, each of which
might serve all trees at once.

`dino_dog_toy`, 4 KiB blocks (32 rows), one cold query, only the tree count
varied. Descriptor accesses equal the check count exactly, so a miss is a block
read and the rest is reuse:

| Trees | Checks | Blocks read | Checks per block | Block reuse |
|-------|--------|------------|-----------------|-------------|
| 1 | 136 | 15 | 9.07 | 89% |
| 2 | 131 | 38 | 3.45 | 71% |
| 4 | 130 | 90 | 1.44 | 31% |
| 8 | 135 | 103 | 1.31 | 24% |

At eight trees nearly every check needs its own block. The trees do converge on
the same near neighbours, but that never turns into block reuse: a descriptor
one tree has evaluated is not evaluated again by another. The overlap shows up
as *fewer checks*, and what remains to be read is the non-overlapping part,
which is scattered. This is why small blocks win, and why a 1,000-query batch
touches 87% of a DinoLedge file's blocks at 64 KiB: 130 scattered reads per
query against 18,950 blocks reaches almost all of them.

[`scripts/kdf_descriptor_orderings.py`](../../../scripts/kdf_descriptor_orderings.py)
compares orderings on one forest. Four trees, 4 KiB blocks, one cold query:

| Policy | `dino` blocks | `dino` reuse | DinoLedge blocks | DinoLedge reuse | DinoLedge batch reads |
|--------|--------------|-------------|-----------------|----------------|----------------------|
| tree-0 leaf order | 90 | 30.8% | 114 | 14.9% | 83,369 |
| Morton over 3 principal components | 123 | 5.4% | 133 | 0.7% | 110,409 |
| greedy co-occurrence packing | 74 | 43.1% | 111 | 17.2% | 86,551 |

A Z-order curve over the top three principal components is worse. Three
components summarize 128 correlated dimensions too loosely: descriptors next to
each other on the curve are often not in the same leaf of any tree. The curve
gives up tree 0's contiguous leaves and gains almost nothing. Greedy
co-occurrence packing visits leaves round-robin across trees and emits each
leaf's unplaced members together. It reads fewer blocks at 694k descriptors
(74 against 90). The gain nearly vanishes at 9.7M (111 against 114), and the
policy reads more blocks over a batch. At 694k descriptors all three policies
read about 22,200 blocks from a corpus of 21,698.

**Decision.** The writer keeps tree-0 leaf order by default. These three
orderings do not set a limit on what other policies could achieve, and a caller
may supply another permutation without changing results.

## Descriptor block size

**Question.** How many descriptors should one compressed block hold?

**Why these measurements.** Once descriptor blocks stopped being one ZIP entry
each, block size no longer affected the number of entries, which had been the
whole cost of small blocks. Two costs remain and pull in opposite directions.
Large blocks decode bytes no query wanted, because reads are scattered. Small
blocks compress worse and need a larger offsets array. A sweep on the largest
corpus shows where the balance lies.

**Code state.** Before the 2026-09-11 access-path revision, with one digest per
block. The open times include parsing those digests (about 303,000 at 4 KiB, or
10 MB of JSON). The format now holds one digest per section, so a repeat would
not pay that cost.

DinoLedge, 4 GiB budget, 1,000-query batch:

| Block | Rows | File | Open | Cold batch | Decoded | Seed amp. |
|-------|------|------|------|-----------|---------|-----------|
| 2 KiB | 16 | 1,247.8 MB | 239 ms | 1.18 s | 466 MiB | 76x |
| 4 KiB | 32 | 1,228.4 MB | 165 ms | 1.45 s | 602 MiB | 89x |
| 8 KiB | 64 | 1,218.5 MB | 145 ms | 1.60 s | 816 MiB | 114x |
| 16 KiB | 128 | 1,213.8 MB | 130 ms | 1.86 s | 1,091 MiB | 164x |
| 64 KiB | 512 | 1,211.6 MB | 113 ms | 2.10 s | 1,458 MiB | 437x |
| 256 KiB | 2,048 | 1,211.9 MB | 139 ms | 2.04 s | 1,407 MiB | 1,407x |

Smaller blocks decode less for the same answers: 466 MiB at 2 KiB against
1,458 MiB at 64 KiB. That outweighs the extra reads, because a read is now a
seek to an offset rather than a directory lookup. On the other side, 2 KiB
frames make the file 34 MB larger than 16 KiB frames do, mostly because zstd
has less context to compress with. A warm batch is 0.07-0.08 s at every size, so
block size matters only for cold and one-shot queries. On this table alone the
best value is 4 to 8 KiB, where the file has grown by well under 1%.

The workload that matters more is whole-image and patch queries under a cache
much smaller than the file. A follow-up therefore measured that case directly,
together with leaf size. Leaf 16 was calibrated to 109 checks (recall@1 0.650)
and leaf 32 to 183 checks (0.652), against 1,000 held-out descriptors from the
9,702,948-descriptor corpus. Fixed: four trees, seed zero, 64 KiB tree chunks
and four query workers. Each measured KDF indexed 9,686,564 descriptors after
withholding two 8,192-descriptor images. At a 256 MiB cache, three
fresh-reader rounds alternated the configuration order. Each round queried those
two images plus four patch descriptor sets of 131–181 features. Values are
medians of the three rounds, with ranges in parentheses. The patch column times
the nearest-neighbour search only, not RANSAC or geometry lookup.

| Block / leaf | Later image | Reads / image | Patch ANN | Reads / patch |
|---|---:|---:|---:|---:|
| **2 KiB / 16** | **4.12 s** (4.03–5.56) | 648,040 | **99 ms** (96–110) | 13,022 |
| 2 KiB / 32 | 4.50 s (4.40–5.01) | 982,852 | 105 ms (97–110) | 20,249 |
| 4 KiB / 16 | 4.62 s (4.08–4.93) | 614,678 | 103 ms (100–108) | 12,334 |
| 4 KiB / 32 | 5.13 s (5.03–5.92) | 924,227 | 122 ms (107–141) | 18,733 |

Neighbour IDs and distances matched across repetitions and block sizes for each
leaf size. The smaller block adds reads but decodes less per descriptor. Leaf
32's larger check budget adds enough reads to cancel its smaller file. The
whole-image result favours 2 KiB / 16 clearly; the patch ranges overlap.
[The per-round measurements](kdf-default-2026-09-12.json) keep the run order,
times, read counts and file sizes.

**Decision.** The writer's default descriptor block is 2 KiB. That choice costs
2.8% more file and roughly twice the open time of the 4-8 KiB range. It was made
because whole-image queries under a pressured cache were fastest at 2 KiB.
Callers can change it.

This default moved twice during measurement. Both times something other than the
layout dominated the result: first a cache whose hit cost grew with its size,
then one ZIP directory record per descriptor block. Re-derive block-size
guidance after any change to how a block is addressed.

## Tree chunk size

**Question.** What decoded size should a tree chunk target?

**Why these measurements.** Seeding a four-tree search costs one subtree miss
per tree, whatever the query. Large chunks decode more on each miss; small
chunks mean more directory entries. Cold-start amplification on `dino_dog_toy`
from 256 KiB to 16 MiB shows how steep that is. A sweep with patch queries
checks whether smaller chunks also help sparse queries, which they should if
descent were the cost.

Cold-start amplification is 96x to 4,373x for tree-local across 256 KiB to
16 MiB, because its chunks carry descriptors. For the shared layout it is 221x
to 576x, since its tree chunks carry no descriptors.

For patch queries, chunk targets from 128 KiB to 4 MiB moved decoded bytes from
19.7 to 21.7 MB and query time not at all beyond noise. The version-1 corpus
prototype counted descriptor bytes the chunks did not store when it weighed
subtrees, so its chunks underfilled their target by roughly 17x. The current
writer counts only stored tree arrays. For this access pattern, decoded volume
depends mainly on how many distinct descriptor blocks the scattered reads
reach, not on the chunks.

**Decision.** A 1 MiB tree-chunk target. Chunk size has little effect on the
shared layout, and the routing working set at 1 MiB is small (see
[the cost model](#the-pre-build-cost-model-2026-09-09)).

## Leaf size

**Question.** Should the forest builder's 16-feature leaf default change for
file-backed use?

**Why these measurements.** Larger leaves put more of a leaf's members in one
block and make the tree arrays smaller. They also examine more descriptors per
visited neighbourhood, so they need a larger check budget for the same recall.
Comparing leaf sizes at equal recall, rather than at equal budget, is the only
fair comparison.
[`scripts/kdf_leaf_size.py`](../../../scripts/kdf_leaf_size.py) finds the
smallest integer check budget that reaches the recall target for each forest
before timing it. An earlier power-of-two budget grid overshot the target
unevenly and made large leaves look more expensive than they are.

570,889 `dino_dog_toy` descriptors, 1,000 held out, four trees, four workers,
4 KiB descriptor blocks, 64 KiB tree chunks, recall@1 >= 0.65. Median of three
forest/query seeds; each seed's time is the median of three fresh-reader
batches:

| Leaf | Exact budget range | File | 64 MiB batch | 256 MiB batch |
|---:|---:|---:|---:|---:|
| 8 | 157–182 | 80.5 MB | 270 ms | 194 ms |
| 16 | 190–221 | 76.4 MB | 234 ms | 155 ms |
| 32 | 241–278 | 73.6 MB | **210 ms** | 135 ms |
| 64 | 345–415 | 72.3 MB | 218 ms | 138 ms |
| 128 | 524–548 | **71.4 MB** | 217 ms | **132 ms** |

At 64 MiB, leaf 32 is fastest and leaf 64 is within 4%. At 256 MiB, leaves
32–128 differ by 6 ms. Leaf 128 saves only 1.2% of file size over leaf 64 and
has the widest timing range under a pressured cache.

Whole-image and patch holdouts, comparing leaf 16 at budget 210 with leaf 64 at
budget 384 (seed-zero calibrated budgets, leaf 64 rounded up from 383):

| Workload | Cache | Leaf 16 | Leaf 64 |
|---|---:|---:|---:|
| Held-out image, later-image median | 16 MiB | **5.12 s** | 5.92 s |
| Held-out image, later-image median | 64 MiB | 1.81 s | **1.42 s** |
| Held-out image, later-image median | 256 MiB | **0.169 s** | 0.265 s |
| Patch constellation, median query | 16 MiB | **231 ms** | 353 ms |
| Patch constellation, median query | 64 MiB | 100 ms | **75 ms** |
| Patch constellation, median query | 256 MiB | **14 ms** | 19 ms |

At 64 MiB, leaf 64's smaller file and tree working set reduce reads enough to
outweigh its larger budget. At 16 MiB, that budget causes more repeated misses.
With the whole file resident, the extra distance work shows.

A 256 MiB cache holds this whole 72–80 MB corpus, but only part of DinoLedge's
1.19–1.26 GB files. Two held-out DinoLedge images and four patch constellations
were therefore measured at the 1,000-query calibration points: leaf 16 at budget
128 (recall 0.657), leaf 32 at 256 (0.679) and leaf 64 at 512 (0.685). The
larger leaves have slightly higher recall, so this brackets the comparison
rather than matching recall exactly:

| Leaf | File | Held-out image | Reads/first image | Patch query | Reads/patch |
|---:|---:|---:|---:|---:|---:|
| 16 | 1,256 MB | **3.77 s** | 677,358 | **89 ms** | 14,636 |
| 32 | 1,210 MB | 6.09 s | 1,208,432 | 134 ms | 25,421 |
| 64 | **1,188 MB** | 11.45 s | 2,252,895 | 236 ms | 45,607 |

With 256 MiB holding about a fifth of the file, reads scale with the check
budget. The 5% file reduction from leaf 16 to 64 does not make up for four times
as many checks. [The recorded measurements](kdf-leaf-size-2026-09-11.json)
include all three small-corpus seeds and both corpora's holdout comparisons.

**Decision.** Leaf 16 stays the general default, and it is the measured choice
for a large index with a 256 MiB cache. Leaves of 32 to 64 are the measured
range when the cache is close to the stored working set, or when file size
matters more than query latency.

## Cache hit cost

**Question.** How should the cache track recency, so that a hit stays cheap as
the cache grows?

**Why these measurements.** The shared layout performs one cache lookup per
descriptor examined, so the cost of a hit multiplies through every query. The
first benchmark found that cost growing with cache size, which would have
decided the layout comparison on a defect.

The first cache promoted an entry on every hit by searching a list of every
resident key. A hit cost 33.5 us on a file holding about 23,000 descriptor
blocks. Stamping entries with a counter made the hit cost flat at about
0.19 us. The cache now keeps an indexed doubly linked list per shard. A
cache-only release probe (`benchmark_cache_churn`, 10,000 replacements, no file
or zstd work) on 2026-09-10 compared the previous cache, whose eviction scanned
the resident entries, with the indexed list:

| Resident entries in one shard | Previous ns/replacement | Indexed LRU ns/replacement |
|---:|---:|---:|
| 1,024 | 8,197 | 457 |
| 16,384 | 136,177 | 346 |
| 65,536 | 1,226,745 | 391 |

These are cache operations, not disk throughput. Across these changes, a
256-query synthetic resident benchmark in the version-1 format went from 6.31 ms to 0.74 ms for
tree-local and from 8.68 ms to 2.29 ms for the shared corpus, with four
workers. The in-memory forest with four workers takes 0.27 ms. The batch made
34,157 checks and no warm reads, with 4,671 cache hits for tree-local and
30,105 for the shared corpus. Addressing the shared corpus has a cost even when
there is no file I/O.

**Decision.** An indexed doubly linked LRU list per shard: a hit and a victim
removal take constant time.

## Query workers and cache sharding

**Question.** Why did adding query workers not speed up a resident batch, and
what fixes it?

**Why these measurements.** Every node and descriptor a query touches takes a
cache lock. One lock for the whole cache would make workers wait on each other.
Comparing one and four workers, each with one shard and with sixteen, separates
lock contention from rayon's per-task overhead on sub-millisecond queries.

Interleaved arms, nine rounds of 20,000 queries against a fully resident
`dino_dog_toy` file, k = 11:

| Cache | Workers | Median | Range |
|-------|---------|--------|-------|
| 1 shard | 1 | 74.2 us | 61.9-86.0 |
| 16 shards | 1 | 72.4 us | 59.6-81.5 |
| 1 shard | 4 | 72.6 us | 69.1-76.9 |
| 16 shards | 4 | **46.8 us** | 45.8-49.1 |

With one worker the arms overlap. With four workers, sharding is 1.55x faster,
and the spread tightens to about +/-3% against +/-15% for the contended arms.
The workers did not help because of lock contention, not because of rayon's
per-task overhead. So sharding fixes it, and reducing task overhead would not.

**Decision.** The cache is split into up to 16 independently locked shards.
How the shard count is capped is in the
[spec](lazy-kdforest-query.md#access-path).

## Reading descriptor blocks

**Question.** How should descriptor blocks be read from the file when several
workers miss at once?

**Why these measurements.** A seek followed by a read shares the file cursor,
so concurrent misses must take a lock. A four-thread run measures what that
lock costs. A one-thread breakdown of each stage of a block read shows which
stages are worth changing.

A four-thread run on 2026-09-11 measured 7.02 microseconds per completed block
with a shared synchronous handle and no cache. With the corrected cache and the
independent handle pool it measured 3.52, and with private handles and no cache
3.13. These are throughput figures with the OS cache warm, not device latency
or an end-to-end speedup.

A one-thread run on a 547,651-descriptor file (32 rows per block, 20,000 sampled
frames, OS-warm reads) timed the stages separately:

| Stage | Mean microseconds/block |
|---|---:|
| Seek, read and frame allocation | 4.04 |
| Explicit-offset read and frame allocation | 1.12 |
| New zstd context and decode | 4.08 |
| Reused zstd context and decode | 3.76 |
| Hash comparison and decoded-byte copy | 0.50 |

These exclude cache admission, traversal and contention between workers. Run the
ignored `profile_corpus_misses` release test with `KDF_PROFILE_PATH` set to a
`u8` file to reproduce them. The test checks that both read methods and both
decoder methods produce identical bytes.

**Decision.** Descriptor and geometry blocks are read with positioned reads, so
no lock is held around a seek. On Windows, positioned reads on one synchronous
handle still serialize, so the corpus opens a pool of independent handles sized
by `query_workers`. Each executing thread reuses one zstd context. Why the
handle pool uses `ReOpenFile` rather than a cloned handle is in the
[spec](lazy-kdforest-query.md#access-path).

## End-to-end held-out images after the access-path revision

**Question.** How much did the 2026-09-11 access-path revision change real image
queries, and how far is the lazy path still from the in-memory forest?

**Why these measurements.** Whole held-out images are the heaviest realistic
query. Three cache sizes cover the range from a cache much smaller than the
file to one that holds it.

82 Dino images (547,651 descriptors) are indexed and three held-out images
(23,238 descriptors) are queried. The run uses four workers for the in-memory
forest and the lazy path, four trees, 16-feature leaves, k = 11, check budget
128, 4 KiB descriptor blocks and 64 KiB tree chunks. Every lazy and reloaded
result is checked against the in-memory indices and distances.

| Decoded cache | Before: later image seconds | After: later image seconds |
|---|---:|---:|
| 16 MiB | 4.72–5.23 | 3.13–3.16 |
| 64 MiB | 1.49–1.51 | 1.23–1.23 |
| 256 MiB | 0.37–0.38 | 0.10–0.13 |

Each endpoint is a run's median over the second and third images, across two
runs per implementation. The baseline is the release extension installed
before the review, not a fresh rebuild. First-image lazy times after the fix
are 2.48–2.54 s, 1.04–1.12 s and 0.25–0.26 s. [Raw measurements and
command](kdf-review-2026-09-11.json) include all per-image times and I/O
counters.

The in-memory forest takes roughly 0.02 s per later image. At 16 MiB the final
image causes approximately 632,700 block reads: 1.99 GB compressed and 2.54 GB
decoded from a 73 MB file, so repeated eviction and decoding explain the gap.
At 256 MiB the same image needs one read and no eviction. The gap that remains
is address validation, row mapping and pin and cache bookkeeping.

**Decision.** The revision is kept: it lowered those costs without changing
candidate order. These results show that the two paths do not cost the same
even with the file resident, which leads to the next sections.

## Cache budget for repeated searches

**Question.** Is the 256 MiB default cache large enough for repeated patch
searches against a large index?

**Why these measurements.** Patch searches are often repeated against one
index. Timing an identical second search shows whether the first search's
working set stayed in the cache.

2026-09-23, a 1,455 MB DinoLedge index built by
[`kdf_constellation_progressive_eval.py`](../../../scripts/kdf_constellation_progressive_eval.py)
(`build-kdf` with its default layout). Eight evenly spaced source images each
supplied 50 descriptors; `k=32`, 512 leaf checks. The median time for an
identical second search was 374 ms at 256 MiB, 10 ms at 512 MiB and 10 ms at
1 GiB. The check reads `.sift` descriptors and the `.kdf` only, and loads no
reconstruction.

**Decision.** The 256 MiB default was not changed; it admits DinoLedge's
largest blocks. Instead the spec tells callers with a large index to compare
repeated-query latency at several budgets before blaming the search. This result
locates the working-set size for this file and query shape only; it is not a
universal default.

## Which workloads suit the file-backed path

**Question.** When is a `.kdf` worth using instead of an in-memory forest?

**Why these measurements.** The answer depends far more on the workload than on
any packing choice. Two real workloads are measured:
- fitting a new image into an indexed capture, which queries thousands of
  descriptors;
- localizing a patch, which queries a few hundred.

[`scripts/kdf_new_image_query.py`](../../../scripts/kdf_new_image_query.py) and
[`scripts/kdf_patch_localize.py`](../../../scripts/kdf_patch_localize.py) take
these measurements. Both return results identical to the in-memory forest at
every cache budget tried.

**Fitting a new image.** Ten images are withheld from the index, spread through
the sequence so that each one's neighbouring frames are still present. At 9.6M
descriptors and 1,186 images:

| Ready by | Time to first answer | Per image after |
|----------|---------------------|-----------------|
| Rebuilding from `.sift` | 13-19 s | 0.37 s |
| Loading the `.kdf` | 32 s | 0.36 s |
| Opening the `.kdf` | **0.22 s** | 0.57 s |

Loading is slower than rebuilding. The median splits take about 9 s, while
loading decompresses the corpus and every tree array and then writes 9.7M
descriptors into feature-ID order, a random write across a 1.24 GB array.
Opening is two orders of magnitude quicker than either. So a `.kdf` wins for a
few arrivals and loses once a rebuild has paid for itself: the crossover is
about three images at 630k descriptors and about fifty at 9.6M.

A whole image is not a sparse query. 8,192 descriptors at a 128-leaf budget make
about a million checks, which reach 279,658 of a DinoLedge file's ~303,000
descriptor blocks (92%). At 256 MiB against that 1.2 GB file an image takes
24 s, with 730,000 evictions. At 4 GiB it takes 0.47 s with none. A cache large
enough to be fast is as large as the corpus.

**Localizing a patch.** Take the features inside a rectangle in one image, look
each one up with `k = 32`, group the hits by image, and keep the images whose
correspondences survive affine RANSAC. A 400-pixel patch is 100-1,400 features.

| Corpus | Query, in memory | Query, `.kdf` | RANSAC | Reads | Decoded |
|--------|-----------------|--------------|--------|-------|---------|
| 696k, 85 images | 1.4 ms | 32 ms | ~433 ms | 2,784 | 15 MB |
| 9.7M, 1,196 images | 7.3 ms | 280 ms | ~2 s | 11,605 | 93 MB |

The index query is about 7% of the end-to-end work at 696k and a similar share
at 9.7M, because geometric verification dominates. The file-backed path's
20-40x overhead on the query costs a few percent of the job, and the index never
has to be rebuilt or held in memory.

**Whole-corpus algorithms.** One query at a 128-check budget makes about 241
cache lookups, 132 of them for descriptors. The in-memory forest answers it in
about 1 us. The file-backed path, fully resident and with every fix above,
takes about 47 us. Four costs came off that path:
- a hit walked an ordered recency list;
- a hit hashed the key twice;
- dropping a pin always called `notify_all`, waking nobody;
- one lock served every access.

What remains is a lock, a hash lookup and an `Arc` clone per descriptor, against
an indexed read. Tuning cannot close that difference. The whole-corpus matcher
in [track-cluster-matching.md](track-cluster-matching.md) runs correctly against
a `.kdf` and returns byte-identical clusters at every scale measured, at tens of
microseconds per query against about 1 us in memory. Its own intermediate arrays
are each `N * (d + 1)` entries in both paths, roughly 850 MB each at 9.7M
descriptors against a 1.24 GB corpus. Taking the index out of memory therefore
removes only about a third of the footprint. At 696k descriptors the two paths
peak within 7% of each other.

**Decision.** The file-backed path is for sparse queries such as patch
localization, and for being ready to answer without a build. For a workload that
revisits most of the corpus, load the forest into memory or give the cache the
whole file. Making a whole-corpus algorithm cheap from a file would need the
descriptors indexed directly with no lock or hash on the path, which is the
memory-mapped flat array that
[kdf-layout-measurements.md](kdf-layout-measurements.md#format-tradeoffs)
describes. Making the matcher itself out-of-core would need its own arrays
streamed.

## Not yet measured

These would settle questions the measurements above leave open:

- **Twenty trees.** A shared corpus avoids nineteen copies rather than three,
  and the cost model above predicts an 87% decoded reduction. A measured file
  size and cold-query time at twenty trees would show whether the four-tree
  defaults still hold.
- **`float32` descriptors.** The format and the query path carry them; the
  Python bindings do not, and no run has measured them.
- **One machine.** Every run above is on one Windows desktop with a local NVMe
  SSD. An HDD or a network filesystem could move the block-size balance.
- **Grouping descriptor reads by block.** Requests to the same block that are
  not consecutive could share one read. This is an experiment in
  [drafts/kdf-shared-descriptor-reads.md](../../drafts/kdf-shared-descriptor-reads.md).
  A useful measurement must include the sorting and replay costs; a lower
  cache-hit count alone does not show an improvement.
