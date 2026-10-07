# KDF Layout Measurements

This file records the measurements that chose the on-disk layout of the
[`.kdf` approximate nearest-neighbor index](../../formats/kdf-file-format.md): how
large a four-tree index over 9.7 million SIFT descriptors is when each tree
stores its own copy of the descriptors (the rejected version-1 tree-local layout)
and when all trees share one descriptor corpus (the layout the format uses), and
why that corpus is compressed in blocks rather than stored as one flat array.
The format spec states the layout; this file holds the evidence for it. The
query-time measurements of the same layouts are in
[lazy-kdforest-query-measurements.md](lazy-kdforest-query-measurements.md#shared-corpus-versus-per-tree-copies).

## Version-1 DinoLedge layout study (2026-09-09)

This section records the measurements that selected the one-corpus
layout. Tree-local paths and totals below describe the rejected version-1
alternative, not entries the current writer emits.

Read-only inspection of `C:\DataSets\DinoLedge\frames` finds one feature set,
`features/sift-sfmtool-3dcd2b2f8c892d12c3ffe28cedce19c9`. All 1,196 images have
a corresponding SIFT file. Counts come from the existing ZIP descriptor shapes
and image metadata; compressed descriptor sizes are the stored zstd frame sizes,
excluding ZIP headers. GB below means decimal bytes / 10^9; MiB means bytes / 2^20.

| Measured source property | Value |
|--------------------------|-------|
| Images | 1,196 JPEGs, each 2160 x 3840 |
| Image files combined | 1,837,792,903 bytes (1.838 GB) |
| SIFT files combined | 1,178,249,456 bytes (1.178 GB) |
| Descriptor rows | 9,702,948, each 128 uint8 values |
| Features per image | Minimum 1,063; median and maximum 8,192 |
| Descriptor bytes, decoded once | 1,241,977,344 bytes (1.242 GB) |
| Existing compressed descriptor frames | 956,433,582 bytes (0.956 GB) |

Assume tree-local descriptor storage, four trees, leaf_size 16, all descriptors
included, and original feature IDs
assigned by lexicographically sorted image filename then original SIFT row.
The current median builder halves each node's feature count, so these shape
calculations do not require knowing the chosen split axes or building the trees.
Each tree has 1,048,576 leaves (9 or 10 features each), and 2,097,151 total nodes.
These are calculated topology counts, not a measured serialized forest.

At a 1 MiB decoded target, each tree has one routing chunk of 2,047 internal
nodes and 2,048 complete-subtree chunks. Each subtree contains 1,023 nodes and
4,737 or 4,738 descriptors, occupying 667,227 or 667,359 decoded bytes. The
next parent is too large, so the target underfills to about 652 KiB. One routing
chunk occupies 83,927 decoded bytes and has no feature rows.

Illustrative layout for a file at the workspace root (chunk 1's exact P depends
on traversal; 4,737 is one of the two valid sizes):

```text
DinoLedge.kdf
  metadata.json.zst                  # N=9702948, D=128, four trees
  images/
    metadata.json.zst                # image_count=1196
    names.json.zst                   # frames/DinoLedge_0001.jpg, ...
    feature_tool_hashes.1196.uint128.zst
    sift_content_hashes.1196.uint128.zst
  origins/0/
    image_indexes.131072.uint32.zst
    image_feature_indexes.131072.uint32.zst
  ...                               # blocks 1 through 73, also 131072 rows
  origins/74/
    image_indexes.3620.uint32.zst
    image_feature_indexes.3620.uint32.zst
  trees/0/chunks/0/                  # upper routing nodes
    nodes.10.2047.uint32.zst
    splits.2047.uint8.zst
    feature_ids.0.uint32.zst
    vectors.0.128.uint8.zst
  trees/0/chunks/1/                  # complete subtree
    nodes.10.1023.uint32.zst
    splits.1023.uint8.zst
    feature_ids.4737.uint32.zst
    vectors.4737.128.uint8.zst
  ...                               # through trees/0/chunks/2048
  ...                               # trees 1, 2, 3 have the same counts
  content_hash.json.zst
```

Directory lines here are prefixes, not ZIP directory entries. There are 8,196
tree chunks, 32,784 tree entries, 150 origin entries and six metadata/image/hash
entries: **32,940 ZIP entries**. Workspace relative_path is `"."`; its absolute
path is `C:\DataSets\DinoLedge`. Image 0 is `frames/DinoLedge_0001.jpg`, and feature
IDs 0 through 8,191 map to image 0, SIFT features 0 through 8,191. Feature ID 8,192
maps to image 1, feature 0. Hashes are copied from each source's hash metadata.
The two image hash arrays together occupy only 38,272 decoded bytes.

## Size budget: measurements versus projection

Four trees' decoded arrays total **5,467,089,308 bytes (5.467 GB)**:
4,967,909,376 vector bytes, 155,247,168 feature-ID bytes, and 343,932,764 node/split
bytes. Origins add 77,623,584 decoded bytes, but compress particularly well in
image/feature order: encoding the actual complete mapping as 75 pairs of zstd
level-3 frames measured **1,408,746 bytes** before ZIP headers. These mappings
are stored once, not four times.

A compression probe decoded every nineteenth SIFT file in filename order
(63 files, 65,224,960 descriptor bytes), then compressed descriptor-only blocks
at zstd level 3 with row counts matching the 1, 8 and 16 MiB layouts.
Image-order bytes compressed to 76.96–76.98% of raw size; a seeded random row
shuffle (Python Random seed 0) compressed to 77.44–77.46%. These are proxy orders,
**not kd-tree leaf order**, and neither is a bound on its compression. The
existing complete corpus's descriptor ratio is about 77.01%.

Applying the measured proxy ratios to four vector copies projects **3.82–3.85 GB
for vectors alone**. Allowing up to the decoded 0.499 GB for ID/node/split arrays
as a conservative planning allowance, plus origins and metadata, gives a useful
rounded planning range of **4.0–4.4 GB (about 3.7–4.1 GiB)** for this `.kdf`.
ZIP headers/directory are only roughly 6–8 MB at 32,940 entries with these names;
chunk metadata/hash JSON adds a few MB decoded before compression. Near or over
4 GiB, writers must enable ZIP64 where individual offsets/sizes require it.

Eight trees roughly double the dominant storage, giving a planning range near
8.0–8.8 GB. The version-1 shared descriptor corpus removes three raw vector copies
(3.726 GB), approximately 2.87–2.89 GB compressed under these proxy ratios.
The format additionally embeds six float32 geometry values per SIFT feature:
232,870,752 decoded bytes for this 9,702,948-row corpus. JPEG pixels and
thumbnails remain external. The geometry's compressed size was not part of this
version-1 measurement.

## The file this projected: measured

Building the projected file confirms the decoded arithmetic exactly and lands at
the bottom of the projected range. Four trees over 9,701,948 of these descriptors
at a 1 MiB chunk target and zstd level 3, via the version-1 revision of
[`scripts/benchmark_kdf_layouts.py`](../../../scripts/benchmark_kdf_layouts.py)
(the current script measures version 3):

| Section | Decoded | Stored | Ratio |
|---------|---------|--------|-------|
| `tree_vectors` | 4.9674 GB | 3.7975 GB | 76.45% |
| `tree_chunks` | 0.4992 GB | 0.2280 GB | 45.69% |
| `metadata` + `content_hash` | 0.0008 GB | 0.0001 GB | — |
| **Payload** | **5.4674 GB** | **4.0257 GB** | **73.63%** |

The decoded column reproduces the counts above to the byte: 5.467 GB total, and
`tree_chunks` holds the 155,247,168 feature-ID bytes and 343,932,764 node and split
bytes together. The file is **4.028 GB** including 2.7 MB of ZIP headers and
directory across 16,394 entries.

Two things the projection could not know. Descriptors compress to **76.45%** in
kd-tree leaf order, marginally better than the 76.96–76.98% image-order proxy —
so the proxy was sound, and the vectors alone came in at 3.7975 GB, just under
the projected 3.82–3.85 GB. And the ID/node/split arrays are far from
incompressible: the ten-column node record compresses to **25.63%**, so those
arrays cost 0.223 GB stored rather than the 0.499 GB the conservative allowance
reserved. Both errors push the same way, which is why the total landed at 4.026 GB
rather than mid-range.

The same forest in the version-1 shared layout is **1.212 GB**, 3.33x smaller: one 0.9489 GB
descriptor corpus at the same 76.41% ratio, plus a 0.0328 GB row map and a 0.0001 GB
offsets array, against four copies. That is 2.816 GB saved, against the
2.87–2.89 GB projected. Its `tree_chunks` section is byte-identical to tree-local's,
which is what makes the comparison a measurement of storage rather than of two
different forests. What the layout costs in query time is measured in
[lazy-kdforest-query-measurements.md](lazy-kdforest-query-measurements.md#shared-corpus-versus-per-tree-copies).

Reproduction: enumerate sorted `features/*/*.sift`; read metadata and hash JSON
with ZIP + zstd; sum descriptor entry shapes and stored frame sizes. Recursively
calculate `nodes(n)=1` for n <= 16, otherwise `1+nodes(floor(n/2))+nodes(ceil(n/2))`;
accept maximal subtrees satisfying `33*nodes(n)+132*n <= target`. Compress the
two little-endian origin columns in blocks of 131,072 rows. The SHA-256 of the
concatenation of `filename + space + stored content_xxh128 + newline` for the
sorted source files is
`2411cdb97d9c93912449ab767d296aa49710c1ad081016d0611b5722e2a409cd`.
This identifies the inspected inventory, not independent verification of all
source payload hashes. No source files were modified and no `.kdf` was built.

## Format tradeoffs

The format pays one uint32 row-map entry per feature to remove `T-1` descriptor
copies. The version-1 comparison measured the corpus layout 3.3-3.4x smaller
across corpora spanning 276x in size and faster in every measured regime except
a fully resident warm batch. That evidence made the corpus layout the only
representation from version 2 on, rather than a caller option.

The grouped integer node columns are a deliberate adaptation of the usual
one-entry-per-column convention: ten separate entries per chunk would cost ten
reads and ten zstd frames to route through a single node. The format allows
arbitrary partitions, so tree-chunk and corpus-block tuning do not require a
version change. Descriptor performance measurements are in
[lazy-kdforest-query-measurements.md](lazy-kdforest-query-measurements.md).

Descriptors are compressed, and therefore blocked and indexed, rather than
stored as one flat fixed-stride array a reader could index arithmetically. The flat
form is genuinely attractive on paper: it deletes the block-size choice, deletes
`features/block_offsets`, and makes a descriptor read exactly `D * w` bytes at
`row * D * w`. It costs about 24% more than the blocked corpus when descriptors
compress to 76.4%.

What makes that trade unattractive is the storage layer's granularity. A filesystem
read is a page, commonly 4 KiB, so a 128-byte descriptor read moves a page anyway;
scattered access pays about one page per descriptor in either form. Compression does
not add a page to that cost — it *removes* pages, by raising how many descriptors a
page holds. The choice is therefore not "fewer bytes versus simpler addressing" but
"fewer pages versus simpler addressing", and blocking wins on the axis that turned
out to dominate.

A flat array remains the better shape for a consumer that memory-maps the corpus and
leaves caching to the operating system, which is a different design rather than a
tuning of this one.
