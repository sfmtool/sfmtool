# The KDF file format

**Status:** Draft. Proposed version 1 for review; no implementation exists.
The proposal supports immutable, self-contained forests with either tree-local
vector copies or one shared vector table. Both layouts are specified for
implementation and comparison; shipping defaults remain benchmark decisions.

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

This proposal uses the [archive container](../formats/archive-container.md).
The companion [lazy query proposal](lazy-kdforest-query.md) defines the query
API and packing policy; all stored values and validity rules are defined here.
The intended standing location is `specs/formats/kdf-file-format.md`.

## Container and entry layout

ZIP entries use STORE, with one independent zstd frame per entry, no external
dictionary, encryption, or multi-volume archives. ZIP64 is supported and
required when ZIP sizes or offsets exceed their ordinary limits. Binary arrays
are little-endian, row-major, without headers. JSON is compact UTF-8. Directory
names are logical prefixes, not separate directory entries. Names are unique.

| Entry | Meaning |
|-------|---------|
| `metadata.json.zst` | Version, dimensions, scalar type, tree and chunk directory |
| `trees/{t}/chunks/{c}/nodes.8.{M}.uint32.zst` | Eight contiguous node columns |
| `trees/{t}/chunks/{c}/splits.{M}.{scalar_type}.zst` | Split coordinate for each node |
| `trees/{t}/chunks/{c}/feature_ids.{P}.uint32.zst` | Original row ID of each locally stored vector |
| `trees/{t}/chunks/{c}/vectors.{P}.{D}.{scalar_type}.zst` | Local vector rows; tree-local layout only |
| `features/storage_rows.{N}.uint32.zst` | Original feature ID to shared storage row; shared layout only |
| `features/blocks/{b}/vectors.{R}.{D}.{scalar_type}.zst` | Shared vector rows; shared layout only |
| `content_hash.json.zst` | Metadata, chunk and whole-file hashes |

In `nodes.8.{M}.uint32.zst`, the eight columns are contiguous rows of a
row-major `(8, M)` array. This uses one entry for all integer node fields to
avoid eight separate reads and frames. Other arrays follow the table literally.
`t` and `c` are zero-based decimal integers without leading zeroes; `M`, `P`
and `D` are decimal counts. Tree-local chunks have all four entries, even when
P = 0. Shared-layout tree chunks have the first three entries and no vectors.

A reader resolves ZIP offsets once from the central directory. Paths do not
imply that the ZIP directory must be rescanned for each node. Physical entry
order does not affect validity. Writers should place each chunk's four entries
consecutively in the order above, for coalesced reads.

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
`32*M + sizeof(scalar_type)*M + 4*P`, plus
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

The eight uint32 columns of `nodes.8.{M}.uint32.zst`, in order, are:

| Column | Internal node | Leaf node |
|--------|---------------|-----------|
| `kind` | Legend code for internal | Legend code for leaf |
| `logical_node_id` | Stable node ID within this tree | Same |
| `split_dimension` | Zero-based axis, less than D | Zero |
| `left_chunk` | Chunk ID within this tree | Zero |
| `left_node` | Local node index in left chunk | Zero |
| `right_chunk` | Chunk ID within this tree | Zero |
| `right_node` | Local node index in right chunk | Zero |
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
ties. The root has logical ID zero. References never cross trees.

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
mapping do not change when storage rows are reordered. Each block is a complete
independent zstd frame. Readers locate a vector by division/remainder of its
storage row by Q. A full verifier checks the map is a permutation and validates
all referenced vectors. Shared block byte length is exactly R*D*scalar width.

Both layouts are introduced in version 1. Their entry sets are mutually
exclusive and readers implementing version 1 support both. Converting layouts
preserves feature IDs, logical node IDs, topology, split values, leaf order,
vector bytes and source mappings; hashes and chunk boundaries may change.

### Hash composition

`content_hash.json.zst` has `metadata_xxh128`, `chunks_xxh128` (an array of
arrays indexed by tree then chunk), and `content_xxh128`. Each value is a
32-character lowercase hexadecimal XXH128 digest, except the nested arrays.
Metadata hashes the exact decoded JSON bytes. A chunk hashes the concatenated
decoded bytes of nodes, splits, feature IDs, and, in tree-local layout, vectors,
in that order. Shared layout additionally requires `storage_rows_xxh128` (the
decoded row map's digest) and `descriptor_blocks_xxh128` (an array of digests
over each block's decoded vector bytes in numeric block order). These fields
are absent in tree-local layout.
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
may reject files exceeding explicit resource limits.

## Implementations

Proposed crate: `sfmtool-kdf-format`, alongside the existing crates under
[crates/](../../crates/), depending on `sfmtool-archive-io` for container
primitives. It owns encoding, structural validation, indexed chunk reading,
writing and full verification, with no dependency on `sfmtool-core`. Core owns
forest construction and queries. There is no Python binding in this proposal.

## Review decisions

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
4,737 or 4,738 descriptors, occupying 659,043 or 659,175 decoded bytes. The
next parent is too large, so the target underfills to about 644 KiB. One routing
chunk occupies 67,551 decoded bytes and has no feature rows.

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
    nodes.8.2047.uint32.zst
    splits.2047.uint8.zst
    feature_ids.0.uint32.zst
    vectors.0.128.uint8.zst
  trees/0/chunks/1/                  # complete subtree
    nodes.8.1023.uint32.zst
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

Four trees' decoded arrays total **5,399,980,476 bytes (5.400 GB)**:
4,967,909,376 vector bytes, 155,247,168 feature-ID bytes, and 276,823,932 node/split
bytes. Origins add 77,623,584 decoded bytes, but compress particularly well in
image/feature order: encoding the actual complete mapping as 75 pairs of zstd
level-3 frames measured **1,408,746 bytes** before ZIP headers. These mappings
are stored once, not four times.

A compression probe decoded every nineteenth SIFT file in filename order
(63 files, 65,224,960 descriptor bytes), then compressed descriptor-only blocks
at zstd level 3 with row counts matching the proposed 1, 8 and 16 MiB layouts.
Image-order bytes compressed to 76.96–76.98% of raw size; a seeded random row
shuffle (Python Random seed 0) compressed to 77.44–77.46%. These are proxy orders,
**not kd-tree leaf order**, and neither is a bound on its compression. The
existing complete corpus's descriptor ratio is about 77.01%.

Applying the measured proxy ratios to four vector copies projects **3.82–3.85 GB
for vectors alone**. Allowing up to the decoded 0.432 GB for ID/node/split arrays
as a conservative planning allowance, plus origins and metadata, gives a useful
rounded planning range of **3.9–4.3 GB (about 3.6–4.0 GiB)** for this `.kdf`.
This is not a produced file size or a guaranteed bound: actual tree-ordered
compression, node compression and JSON serialization remain unmeasured.
ZIP headers/directory are only roughly 6–8 MB at 32,940 entries with these names;
chunk metadata/hash JSON adds a few MB decoded before compression. Near or over
4 GiB, writers must enable ZIP64 where individual offsets/sizes require it.

Eight trees roughly double the dominant storage, giving a planning range near
7.8–8.6 GB. A shared descriptor corpus would remove three raw vector copies
(3.726 GB), approximately 2.87–2.89 GB compressed under these proxy ratios,
but its random descriptor reads are precisely the tradeoff to benchmark.
JPEG pixels, SIFT keypoints, affine shapes and thumbnails are not embedded.

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

### Remaining format decisions

The principal cost is T copies of the descriptor corpus, one per tree. For
one million 128-byte vectors and four trees that is 512 MB of uncompressed
vector bytes, before IDs, nodes and compression. A single shared corpus saves
space but may require many extra chunks for one leaf. The benchmark in the
companion proposal compares both before freezing version 1.

The grouped integer node columns are a deliberate adaptation of the usual
one-entry-per-column convention. Review whether fewer entry reads justify it.
The format allows arbitrary partitions and supports both descriptor layouts,
so size tuning and layout selection do not require a version change. Shared
descriptor performance estimates are in the companion query draft.
