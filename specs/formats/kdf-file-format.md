# The KDF file format

Version 1 supports immutable, self-contained forests with either tree-local
vector copies or one shared vector table. The shared layout is the measured
general recommendation; callers still choose explicitly because a fully resident
working set retains a real reason to use tree-local storage.

A `.kdf` file stores a set of fixed-width vectors and several binary spatial
partition trees over that set. It supports approximate nearest-neighbor lookup
without loading all vectors or trees into memory. Returned feature IDs identify
rows in the original input, even though each tree stores them in a different
order. Vectors may be byte descriptors such as SIFT, or finite float32 data;
the metric is squared Euclidean distance.

Here, a **feature** is one indexed vector row, including generic non-image
vectors. `feature_count` counts those rows, and a **feature ID** is its zero-based
row ID in the original input. An **image feature** is a feature in a source
image's SIFT file, identified by `(image_index, image_feature_index)`.
`image_feature_indexes` stores those source-file row indices; the origins table
maps each corpus feature ID to its image feature.
The file contains no reconstructed 3D points.

The format uses the [archive container](archive-container.md). The companion
[lazy query design](../core/features/lazy-kdforest-query.md) defines the query
API and packing policy; all stored values and validity rules are defined here.

## Container and entry layout

ZIP entries use STORE, no external dictionary, encryption, or multi-volume
archives. ZIP64 is supported and required when ZIP sizes or offsets exceed their
ordinary limits. Binary arrays are little-endian, row-major, without headers.
JSON is compact UTF-8. Directory names are logical prefixes, not separate
directory entries. Names are unique.

| Entry | Meaning |
|-------|---------|
| `metadata.json.zst` | Version, dimensions, scalar type, tree and chunk directory |
| `trees/{t}/chunks/{c}/chunk.{M}.{P}.{scalar_type}.zst` | One chunk's node columns, splits and feature IDs |
| `trees/{t}/chunks/{c}/vectors.{P}.{D}.{scalar_type}.zst` | Local vector rows; tree-local layout only |
| `features/storage_rows.{N}.uint32.zst` | Original feature ID to shared storage row; shared layout only |
| `features/corpus.{N}.{D}.{scalar_type}.frames` | The shared vector corpus, one zstd frame per block; shared layout only |
| `features/block_offsets.{B}.uint64.zst` | Where each corpus frame starts; shared layout only |
| `content_hash.json.zst` | Metadata, chunk and whole-file hashes |

Every entry is a single zstd frame **except** `features/corpus`, which is a
concatenation of one frame per descriptor block and is named `.frames` to say so.
[Where this format departs from the container conventions](#where-this-format-departs-from-the-container-conventions)
explains both that and the grouped chunk entry.

A chunk entry holds, concatenated in this order and with no padding or header
between them:

1. ten node columns, a row-major `(10, M)` `uint32` array;
2. one split coordinate per node, `M` values of the scalar type;
3. the original row ID of each locally stored vector, `P` `uint32` values.

So it decodes to exactly `40*M + w*M + 4*P` bytes for scalar width `w`, and a
reader rejects an entry that decodes to any other length. A chunk's vectors are a
*separate* entry, present only in tree-local layout and decoding to `w*P*D` bytes;
the reasoning for that split is in
[Where this format departs from the container conventions](#where-this-format-departs-from-the-container-conventions).

`t` and `c` are zero-based decimal integers without leading zeroes; `M`, `P`, `D`
and `B` are decimal counts. Every chunk has exactly one `chunk` entry, even when
P = 0, and in tree-local layout exactly one `vectors` entry beside it.

A reader resolves ZIP offsets once from the central directory. Paths do not
imply that the ZIP directory must be rescanned for each node. Physical entry
order does not affect validity.

## Metadata and versioning

Required fields in `metadata.json.zst`:

| Field | Type and meaning |
|-------|------------------|
| `format` | String `"kdf"` |
| `version` | Integer `1` |
| `scalar_type` | String `"uint8"` or `"float32"` |
| `metric` | String `"squared_l2"` |
| `feature_count` | Integer N, `0 <= N <= 2^32 - 1`; valid IDs are `0..N` exclusive |
| `dimension` | Integer D, `1 <= D <= 65535` |
| `node_kinds` | Canonical writer legend `["internal", "leaf"]`; readers resolve codes through this array |
| `target_chunk_bytes` | Positive integer, writer's target decoded byte size; advisory, not a reader allocation limit |
| `trees` | Nonempty ordered array of tree objects, described below |
| `feature_source` | String `"sift_files"` for mapped descriptors, or `"none"` for generic vectors |
| `descriptor_storage` | String `"tree_local"` or `"shared"`; independent of feature_source |

Each tree has `root: [chunk_id, local_node_index]` (or `null` for N = 0),
and `chunks`, an array in chunk-ID order. Each chunk object has integer
`node_count` M, `feature_count` P, and `decoded_bytes`. The latter equals
`40*M + sizeof(scalar_type)*M + 4*P`, plus
`sizeof(scalar_type)*P*D` in tree-local layout.
Chunk and local node indices fit uint32. M is positive; an empty forest has
no chunks in any tree. No tree is empty when N is positive.

Optional `provenance` is a JSON object of uninterpreted producer information,
such as build parameters, seed, descriptor normalization, or source identity.
It confers no source-file dependency and is not needed to interpret the index.
There is no implicit SIFT normalization or distance conversion.

All entries and fields above are introduced in version 1. Readers reject other
versions and unsupported scalar/metric/kind/source/storage names. Unknown JSON fields are
ignored. Source entries below are conditional; unexpected ZIP entries are
rejected. Changes to binary interpretation, required entries, or required
semantics require a version increment. An older reader rejects rather than
partially interpreting such a file.

## SIFT references and descriptor origins

For `feature_source = "sift_files"`, the following entries and metadata fields
are required. They are absent for `"none"`; mixed mapped/unmapped corpora are
not supported. This adopts the image-table names, hash encodings and reference meanings of
[the SFMR format](../formats/sfmr-file-format.md), without importing its cameras,
poses, thumbnails, reconstructed points or observation-track ordering.

The origin column is named `image_feature_indexes` rather than SFMR's
`tracks/feature_indexes` to distinguish source-image features from corpus
features. Its values have exactly the same meaning as the SFMR column; conversion
preserves the indices without renumbering.

| Entry | Decoded contents |
|-------|------------------|
| `images/metadata.json.zst` | Object with `image_count` I, a nonnegative uint32-sized integer |
| `images/names.json.zst` | Array of I unique workspace-relative POSIX image paths |
| `images/feature_tool_hashes.{I}.uint128.zst` | Per-image feature-tool identity hashes, 16 little-endian bytes each |
| `images/sift_content_hashes.{I}.uint128.zst` | Per-image SIFT content hashes, 16 little-endian bytes each |
| `origins/{b}/image_indexes.{R}.uint32.zst` | Image index for each original descriptor row in this block |
| `origins/{b}/image_feature_indexes.{R}.uint32.zst` | Zero-based feature row in that image's referenced SIFT file |

Metadata additionally has positive integer `origin_block_rows` B. Block b covers
original feature IDs `[b*B, min((b+1)*B, N))`, R is that range's length, and b ranges
from zero through `ceil(N/B)-1`, using decimal names without leading zeroes.
There are no blocks when N = 0. The mapping is stored once, independent of trees,
in original descriptor ID order. Image indices are less than I. Feature indices
refer to the original SIFT row, not its position in a filtered subset or leaf.
Each `(image_index, image_feature_index)` pair occurs at most once. Subsets are valid;
image rows with no indexed features are permitted. An indexed feature maps to exactly one source SIFT
feature, unlike a reconstructed point's multiple SFMR observations.

Required `metadata.workspace` contains `absolute_path` (workspace location at
save time), `relative_path` (POSIX path from the KDF parent to that workspace),
and `contents` (embedded workspace configuration). Contents includes string
`feature_tool`, string `feature_type`, object `feature_options`, and relative
POSIX `feature_prefix_dir`. Resolve the workspace by trying relative_path first,
then absolute_path, then a containing workspace, matching SFMR. Image paths
resolve against that workspace, never directly against the KDF's parent.
For image `frames/a.jpg` and prefix `features/sift-example`, its source is
`frames/features/sift-example/a.jpg.sift` within the workspace. All images use
this one prefix convention; multiple workspace configurations are outside version 1.

The tool hash equals the referenced SIFT file's stored `feature_tool_xxh128`;
the content hash equals its stored `content_xxh128`. Neither hashes the compressed
ZIP file. Source verification checks both identities, feature index bounds,
dimension/type compatibility, and byte equality with the indexed descriptor.
Version 1 stores source descriptors unchanged; transformed embeddings use
`feature_source = "none"` until a transform provenance contract is defined.
Missing or changed SIFT files do not prevent ANN or reading origins: references
identify provenance, while vectors are embedded. External source verification
is separate from verifying the KDF itself.

All source fields and entries are introduced in version 1. They participate in
integrity hashing below. No source file is opened by a normal query.

## Nodes, leaves, and identity

The ten uint32 columns of `nodes.10.{M}.uint32.zst`, in order, are:

| Column | Internal node | Leaf node |
|--------|---------------|-----------|
| `kind` | Legend code for internal | Legend code for leaf |
| `logical_node_id` | Stable node ID within this tree | Same |
| `split_dimension` | Zero-based axis, less than D | Zero |
| `left_chunk` | Chunk ID within this tree | Zero |
| `left_node` | Local node index in left chunk | Zero |
| `left_logical_node_id` | Logical ID of the left child | Zero |
| `right_chunk` | Chunk ID within this tree | Zero |
| `right_node` | Local node index in right chunk | Zero |
| `right_logical_node_id` | Logical ID of the right child | Zero |
| `leaf_start` | Zero | Start row in this chunk's feature-ID/vector arrays |

Leaf lengths are inferred from consecutive leaf starts in **local node order**:
the next leaf start minus this start, with P as the end of the last leaf.
The first leaf starts at zero; lengths are positive. Chunks without leaves
have P = 0. `splits` holds an internal node's split coordinate, and zero for
a leaf. All float32 values are finite; zero in unused float fields is positive
zero. A byte split is an unsigned byte, not a quantized floating value.

Logical node IDs are unique and dense from zero through the tree's node count
minus one. They preserve the source forest's node identity during repacking;
they are independent of disk address and available for deterministic traversal
ties. Each child reference repeats the addressed child's logical ID, so a lazy
query can enqueue a far child with the correct tie key without loading that
child's chunk. The repeated value must match the addressed node. The root has
logical ID zero. References never cross trees.

Every node is reachable exactly once from its tree's root: no cycles, shared
children, unreachable nodes, or duplicate child references. Every leaf owns
one complete range in its chunk; no leaf spans chunks. Every feature ID occurs
exactly once per tree, and copies of the same ID across trees have identical
vector bytes in tree-local layout; in shared layout they reference the same
shared row. IDs are original input rows, not tree-local permutation offsets.
The value `2^32 - 1` is never a feature ID and remains available as query padding.

Every vector below an internal node's left child has coordinate <= its split
on the split dimension; below the right child it is >= the split. Values equal
to a split may occur on either side. The trees need not be balanced and the
format imposes no particular construction algorithm or random generator.

## Chunk independence and integrity

A tree-local chunk contains enough information to evaluate all its leaves.
A shared-layout tree chunk contains leaf IDs whose vectors reside in shared
blocks in the same archive. Neither layout needs an external descriptor corpus.
Internal children may reference another chunk.
Splitting or merging chunks preserves topology, logical node IDs, leaf member
order, and original IDs. Chunk boundaries are not ANN approximation boundaries.
Target size is a decoded-byte packing target, not compressed ZIP bytes. Writers
may include virtual vector bytes when partitioning shared-layout trees to keep
their partitions comparable to tree-local layout; decoded_bytes always records
the actual arrays present. Shared descriptor block sizes are defined separately.
It is a target rather than a bound: a leaf larger than the target is indivisible.

### Shared descriptor addressing

Shared layout requires positive integer `metadata.descriptor_block_rows` Q.
It is absent in tree-local layout, as are all `features/` entries. The storage
row table is a permutation of `0..N`: entry i gives the physical vector row for
original feature ID i. Block b stores rows `[b*Q, min((b+1)*Q,N))`, with R equal
to that interval's length. Blocks are dense from zero through `ceil(N/Q)-1`,
using decimal names without leading zeroes. N = 0 has an empty storage-row entry
and no descriptor blocks. Vectors have the same scalar type and D as the forest.
Each vector occurs once in this table; different IDs with identical vector bytes
remain distinct rows. This is deduplication across trees, not across identities.

The row permutation is explicit, not inferred from tree topology. A writer may
choose any order; readers use the stored map. Leaf membership/order and origin
mapping do not change when storage rows are reordered. Readers locate a vector by
division/remainder of its storage row by Q. A full verifier checks the map is a
permutation and validates all referenced vectors. Block b decodes to exactly
R*D*scalar width bytes.

Every block is a complete independent zstd frame, and all B = `ceil(N/Q)` of them
are stored back to back in the single `features/corpus` entry, in block order.
`features/block_offsets` holds B+1 `uint64` values: block b occupies the stored
bytes `[offsets[b], offsets[b+1])` of that entry, measured from its first stored
byte. `offsets[0]` is zero, the values never decrease, and `offsets[B]` equals the
entry's stored length — a reader checks all three at open, because a truncated or
reordered offsets array would otherwise surface as an unexplained decode failure
on whichever block happened to be read first. Reading a block is therefore a seek
to a recorded offset followed by decoding one frame; nothing outside that range is
read or decoded. N = 0 has an empty storage-row entry, an empty corpus, and
`block_offsets` holding the single value zero.

Both layouts are introduced in version 1. Their entry sets are mutually
exclusive and readers implementing version 1 support both. Converting layouts
preserves feature IDs, logical node IDs, topology, split values, leaf order,
vector bytes and source mappings; hashes and chunk boundaries may change.

### Hash composition

`content_hash.json.zst` has `metadata_xxh128`, `chunks_xxh128` (an array of
arrays indexed by tree then chunk), and `content_xxh128`. Each value is a
32-character lowercase hexadecimal XXH128 digest, except the nested arrays.
Metadata hashes the exact decoded JSON bytes. A chunk hashes its `chunk` entry's
decoded bytes followed, in tree-local layout, by its `vectors` entry's — which is
the same value as hashing nodes, splits, feature IDs and then vectors in sequence,
since the first three are concatenated in that order inside the one entry. Shared layout additionally requires `storage_rows_xxh128` (the
decoded row map's digest) and `descriptor_blocks_xxh128` (an array of digests
over each block's decoded vector bytes in numeric block order — per block, not
over the container entry that holds them, so a digest identifies the same bytes
whatever entry they are packed into). These fields are absent in tree-local
layout. `features/block_offsets` is covered by no digest of its own: it is
addressing rather than content, and a wrong value is caught by the three
structural checks on it plus the block digest of whatever it addressed.
SIFT mode additionally requires `images_xxh128` (one digest over the four images
entries' decoded bytes in lexicographic path order) and `origins_xxh128` (an
array of block digests, each hashing image_indexes then image_feature_indexes decoded
bytes). Both fields are absent in generic mode. The whole-file hash hashes the
metadata digest, then images and origin-block digests if present (numeric block
order), then the storage-row and descriptor-block digests in shared layout,
followed by every tree chunk digest
in numeric tree/chunk order, each serialized as 16 big-endian bytes, following
the archive-container convention. The hash entry itself is excluded.

This identity depends on packing, node layout and JSON serialization, but not
compression level. It is not a canonical identity of the vector set. Repacking
requires recomputing hashes. Hashes detect corruption, not malicious tampering.
ZIP CRC also covers each entry's stored compressed bytes.

Opening can validate the metadata, hash directory, expected entry set, lengths
and references' declared ranges without reading every chunk. Chunk decoding
checks exact sizes, local field constraints and its digest. Full verification
also checks reachability, ID permutations, cross-tree vector equality and split
constraints; it necessarily reads the whole file. Lazy access does not certify
unread chunks. All size arithmetic is checked before allocation, and readers
may reject files exceeding explicit resource limits. The ZIP dependency indexes
the central directory by filename, so opening separately counts its raw file
records and rejects a count greater than the unique-name index; otherwise a
duplicate name would be collapsed before entry-set validation could see it.

## Where this format departs from the container conventions

The [archive container](archive-container.md) sets conventions all the formats in
this repository follow, and two of them are about what an entry is: **one entry
per field, one primitive type per entry**, and **each entry's stored bytes are a
single zstd frame**, with the shape in the name so a reader knows the exact
decoded length before decompressing. Those conventions buy something real — a
consumer reads the columns it wants and skips the rest, and `unzip -l` plus a
`zstd -d` explains the file without this spec.

This format breaks both, deliberately, in two places. The reason is that `.kdf`
is the only format here whose entry count scales with the *data*, rather than
with the schema.

A `.sfmr`, `.sift`, `.matches` or `.camrig` file has a fixed set of entries: one
per column the schema defines, however large the reconstruction. A `.kdf` has one
per tree chunk and one per descriptor block, so a 9.7M-descriptor corpus reaches
tens or hundreds of thousands of entries. At that scale the conventions stop being
free:

- A ZIP entry costs about 164 bytes of local header and central directory record
  with names of this length, and a reader parses the whole central directory
  before it can do anything. That is linear in entry count at roughly 5.6 us per
  entry: 131,260 entries cost 735 ms of open latency, paid before the first query.
- Independent zstd frames do not share compression context, so cutting the same
  bytes into more entries compresses them slightly worse.

Both departures buy back that cost without changing what is stored.

**A chunk's three integer arrays share one entry.** Node columns, splits and
feature IDs are always read together — decoding a node needs the columns and the
splits, reaching a leaf needs the IDs — so the "read the columns you want" benefit
never applied to them. The name still carries every count, so the decoded length is
still known before decompressing and still checked; what is given up is one
primitive type per entry, and the section above says what lies at which offset
instead. This is the same trade the format already makes for the ten node columns,
one level up.

Grouping is not quite free, and the measurement says so: those three arrays
compress to 25.6%, 64.9% and 84.6% separately, and sharing one frame gives 45.7%
overall — 2.4% more stored bytes than the three frames cost, or 5.3 MB on a 4 GB
file. It is paid for by halving the entry count and the open latency with it.

**A chunk's vectors do not join them**, even though a tree-local chunk always reads
its vectors along with its topology. Folding them in was tried and reverted: bulk
descriptor bytes compress at 76%, the integer columns at 46%, and one zstd frame
containing both compresses each worse than two frames do. On DinoLedge it cost 0.7%
of total file size — 28 MB — to save one read per chunk. So the grouping rule this
format follows is narrower than "always read together": group fields that are
always read together **and** compress alike.

**The descriptor corpus is one entry of many frames.** Blocks must stay
independently decodable — reading one descriptor may not require decompressing the
corpus — so they remain one frame each, and each keeps its own digest. What changes
is that the frames are concatenated into a single entry and addressed by a stored
offsets array instead of by the ZIP directory.

This is the departure that matters most, because the useful configurations use
small blocks: at 16 KiB on DinoLedge, blocks were ~76,000 of ~100,000 entries and
are now two, taking open latency from 674 ms to 130 ms. It also makes block size
free in entry count, which it was not before — the choice is now purely about
read granularity. Unlike the grouping above it costs nothing in compression,
because the frames are unchanged; only their addressing moved.

The entry is named `.frames` rather than `.zst` so that a tool which assumes one
frame per entry fails honestly instead of decoding only the first block and
reporting success.

What is preserved is the part that carries the weight. The file is still a ZIP of
STORE entries, so standard tools still list it and still extract any entry. Binary
arrays are still little-endian and row-major with no headers of their own. Names
still encode shape and type, and every decoded length is still derivable from a
name and checked against it. The digests still cover decoded bytes, still compose
into `content_xxh128` the same way, and still identify the same byte sequences they
did when those bytes sat in separate entries. A reader that knows this section can
still be written against the spec alone, in another language, without consulting
the implementation — which is the standard `specs/formats/` is actually held to.

The one genuine loss is shell-level inspection of the corpus: `unzip` will hand
over `features/corpus...frames`, and splitting it into blocks needs the offsets
array rather than a `zstd -d`. The tradeoff was accepted because the alternative is
a file whose open cost grows without bound in the number of descriptors.

## Implementations

The [`sfmtool-kdf-format`](../../crates/sfmtool-kdf-format/) crate sits alongside
the other format crates under [crates/](../../crates/) and depends on
`sfmtool-archive-io` for container primitives. It owns encoding, structural
validation, indexed chunk reading, writing and full verification, with no
dependency on `sfmtool-core`. Core owns forest construction and queries, in
[`features/kdforest/persistent.rs`](../../crates/sfmtool-core/src/features/kdforest/persistent.rs).
The `uint8` half of both is bound for Python on the `sfmtool.spatial`
submodule, in
[`spatial/kdf.rs`](../../crates/sfmtool-py/src/spatial/kdf.rs), so the layout
comparison below can be run without writing Rust. `float32` is not bound: the
eager `KdForest` it would be compared against is `uint8` only.

## Sizing and tradeoffs

### DinoLedge case study (2026-09-09)

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

### Size budget: measurements versus projection

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
8.0–8.8 GB. A shared descriptor corpus removes three raw vector copies
(3.726 GB), approximately 2.87–2.89 GB compressed under these proxy ratios.
JPEG pixels, SIFT keypoints, affine shapes and thumbnails are not embedded.

### The file this projected: measured

Building the projected file confirms the decoded arithmetic exactly and lands at
the bottom of the projected range. Four trees over 9,701,948 of these descriptors
at a 1 MiB chunk target and zstd level 3, via
[`scripts/benchmark_kdf_layouts.py`](../../scripts/benchmark_kdf_layouts.py):

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

The same forest in the shared layout is **1.212 GB**, 3.33x smaller: one 0.9489 GB
descriptor corpus at the same 76.41% ratio, plus a 0.0328 GB row map and a 0.0001 GB
offsets array, against four copies. That is 2.816 GB saved, against the
2.87–2.89 GB projected. Its `tree_chunks` section is byte-identical to tree-local's,
which is what makes the comparison a measurement of storage rather than of two
different forests. What the layout costs in query time is measured in
[lazy-kdforest-query.md](../core/features/lazy-kdforest-query.md#what-the-measurements-found).

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

### Format tradeoffs

The principal cost is T copies of the descriptor corpus, one per tree. For
one million 128-byte vectors and four trees that is 512 MB of uncompressed
vector bytes, before IDs, nodes and compression. A single shared corpus saves
space but may require many extra chunks for one leaf. The benchmark in the
companion query design has measured both layouts: the shared corpus is 3.3-3.4x
smaller across corpora spanning 276x in size, and is faster in every regime
measured except a fully resident warm batch. Because the format carries both,
choosing between them does not change the version-1 wire contract.

The grouped integer node columns are a deliberate adaptation of the usual
one-entry-per-column convention: ten separate entries per chunk would cost ten
reads and ten zstd frames to route through a single node. The format allows
arbitrary partitions and supports both descriptor layouts, so size tuning and
layout selection do not require a version change. Shared descriptor performance
estimates are in the companion query design.

Shared descriptors are compressed, and therefore blocked and indexed, rather than
stored as one flat fixed-stride array a reader could index arithmetically. The flat
form is genuinely attractive on paper: it deletes the block-size choice, deletes
`features/block_offsets`, and makes a descriptor read exactly `D * w` bytes at
`row * D * w`. It costs 24% of a shared file, descriptors compressing to 76.4%.

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

### Writer and summary working memory

The writer streams the shared corpus's compressed frames into its ZIP entry and
retains the offsets and hashes; it does not buffer the complete compressed corpus.
The summary derives array sizes from entry shapes, including the uint64 block
offsets. JSON decompression is bounded by `max_metadata_bytes`; the ZIP STORE size
alone does not bound the expanded JSON. Array-shape arithmetic checks overflow.
