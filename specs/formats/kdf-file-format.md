# The KDF file format

A `.kdf` file is an approximate nearest-neighbor index: a set of fixed-width
vectors, such as SIFT descriptors, plus one or more binary spatial-partition
trees over them, laid out so a reader can answer queries without loading all
vectors or trees into memory. Returned feature IDs identify rows in the original
input, even though each tree stores them in a different order. Vectors may be
byte descriptors such as SIFT, or finite float32 data; the metric is squared
Euclidean distance.

Every vector is stored exactly once, in a blocked corpus shared by all trees.
SIFT-backed files also store each feature's image-space keypoint and 2x2 affine
shape in a geometry corpus blocked the same way, so descriptor matches can be
turned into image constellations without reopening the source `.sift` files.

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
| `features/storage_rows.{N}.uint32.zst` | Original feature ID to corpus storage row |
| `features/corpus.{N}.{D}.{scalar_type}.frames` | Vector corpus, one zstd frame per block |
| `features/block_offsets.{C+1}.uint64.zst` | Where each vector frame starts |
| `features/geometry.{N}.3.2.float32.frames` | SIFT keypoints and affine shapes, one frame per vector block; SIFT mode only |
| `features/geometry_block_offsets.{C+1}.uint64.zst` | Where each geometry frame starts; SIFT mode only |
| `content_hash.json.zst` | One digest per section, and one over all of them |

Here C = `ceil(N / descriptor_block_rows)` is the number of descriptor blocks.
Every entry is a single zstd frame **except** `features/corpus` and optional
`features/geometry`, which concatenate one frame per descriptor block and are
named `.frames` to say so.
[Where this format departs from the container conventions](#where-this-format-departs-from-the-container-conventions)
explains both that and the grouped chunk entry.

A chunk entry holds, concatenated in this order and with no padding or header
between them:

1. ten node columns, a row-major `(10, M)` `uint32` array;
2. one split coordinate per node, `M` values of the scalar type;
3. the original row ID of each locally stored vector, `P` `uint32` values.

So it decodes to exactly `40*M + w*M + 4*P` bytes for scalar width `w`, and a
reader rejects an entry that decodes to any other length. Tree chunks never
contain vector or geometry rows.

`t` and `c` are zero-based decimal integers without leading zeroes; `M`, `P`, `D`
and `C` are decimal counts. Every chunk has exactly one `chunk` entry, even when
P = 0.

A reader resolves ZIP offsets once from the central directory. Paths do not
imply that the ZIP directory must be rescanned for each node. Physical entry
order does not affect validity.

## Metadata and versioning

Required fields in `metadata.json.zst`:

| Field | Type and meaning |
|-------|------------------|
| `format` | String `"kdf"` |
| `version` | Integer `3` |
| `scalar_type` | String `"uint8"` or `"float32"` |
| `metric` | String `"squared_l2"` |
| `feature_count` | Integer N, `0 <= N <= 2^32 - 1`; valid IDs are `0..N` exclusive |
| `dimension` | Integer D, `1 <= D <= 65535` |
| `node_kinds` | Exactly `["internal", "leaf"]`, so node kind code 0 is internal and 1 is leaf; readers reject any other array |
| `target_chunk_bytes` | Positive integer, writer's target decoded byte size; advisory, not a reader allocation limit |
| `trees` | Nonempty ordered array of tree objects, described below |
| `feature_source` | String `"sift_files"` for mapped descriptors, or `"none"` for generic vectors |
| `descriptor_block_rows` | Positive integer Q, the row count of every full vector and geometry block |

Each tree has `root: [chunk_id, local_node_index]` and `chunks`, an array in
chunk-ID order. When N is positive the root is `[0, 0]`, the first node of chunk
0, and readers reject any other address; when N = 0 it is `null`. Each chunk
object has integer `node_count` M, `feature_count` P, and `decoded_bytes`. The
latter equals `40*M + sizeof(scalar_type)*M + 4*P`.
Chunk and local node indices fit uint32. M is positive; an empty forest has
no chunks in any tree. No tree is empty when N is positive.

Optional `provenance` is a JSON object of uninterpreted producer information,
such as build parameters, seed, descriptor normalization, or source identity.
It confers no source-file dependency and is not needed to interpret the index.
There is no implicit SIFT normalization or distance conversion.

Readers reject unsupported scalar, metric, kind, or source names. Unknown JSON fields are
ignored. Source entries below are conditional; unexpected ZIP entries are
rejected. Changes to binary interpretation, required entries, or required
semantics require a version increment. An older reader rejects rather than
partially interpreting such a file.

## SIFT references and descriptor origins

For `feature_source = "sift_files"`, the following entries and metadata fields
are required. They are absent for `"none"`; mixed mapped/unmapped corpora are
not supported. SIFT mode requires 128-D uint8 descriptors. This adopts the
image-table names, hash encodings and reference meanings of
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
| `features/geometry.{N}.3.2.float32.frames` | Keypoint centers and affine shapes in corpus storage order |
| `features/geometry_block_offsets.{C+1}.uint64.zst` | Geometry frame boundaries |

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
this one prefix convention; multiple workspace configurations are outside this format.

One geometry row is the row-major float32 matrix
`[[x, y], [a11, a12], [a21, a22]]`. Its first row is the SIFT keypoint center in
image pixels and its last two rows are the 2x2 affine shape exactly as stored by
the SIFT format. All six values are finite. Geometry rows use the descriptor
corpus's storage permutation and Q-row block boundaries: geometry block b and
descriptor block b therefore cover the same feature IDs in the same row order.
Their compressed frame offsets are independent because the two payloads have
different sizes and compression ratios.

The tool hash equals the referenced SIFT file's stored `feature_tool_xxh128`;
the content hash equals its stored `content_xxh128`. Neither hashes the compressed
ZIP file. Source verification checks both identities, feature index bounds,
dimension/type compatibility, byte equality with the indexed descriptor, and
bitwise equality of the embedded keypoint/affine float32 row. Source
descriptors and geometry are stored unchanged; transformed embeddings use
`feature_source = "none"` until a transform provenance contract is defined.
Missing or changed SIFT files do not prevent ANN or reading origins: references
identify provenance, while vectors are embedded. External source verification
is separate from verifying the KDF itself.

All source fields and entries participate in
integrity hashing below. No source file is opened by a normal query.

## Nodes, leaves, and identity

The `chunk` entry's ten uint32 node columns, in order, are:

| Column | Internal node | Leaf node |
|--------|---------------|-----------|
| `kind` | 0 | 1 |
| `logical_node_id` | Stable node ID within this tree | Same |
| `split_dimension` | Zero-based axis, less than D | Zero |
| `left_chunk` | Chunk ID within this tree | Zero |
| `left_node` | Local node index in left chunk | Zero |
| `left_logical_node_id` | Logical ID of the left child | Zero |
| `right_chunk` | Chunk ID within this tree | Zero |
| `right_node` | Local node index in right chunk | Zero |
| `right_logical_node_id` | Logical ID of the right child | Zero |
| `leaf_start` | Zero | Start row in this chunk's feature-ID array |

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
exactly once per tree, and every occurrence resolves through the one corpus row
map. IDs are original input rows, not storage permutation offsets.
The value `2^32 - 1` is never a feature ID and remains available as query padding.

Every vector below an internal node's left child has coordinate <= its split
on the split dimension; below the right child it is >= the split. Values equal
to a split may occur on either side. The trees need not be balanced and the
format imposes no particular construction algorithm or random generator.

## Chunk independence and integrity

A tree chunk contains topology and leaf feature IDs; its vectors reside in the
blocked corpus in the same archive. Internal children may reference another chunk.
Splitting or merging chunks preserves topology, logical node IDs, leaf member
order, and original IDs. Chunk boundaries are not ANN approximation boundaries.
Target size is a decoded-byte packing target, not compressed ZIP bytes and
counts only the arrays actually present. Descriptor block sizes are defined separately.
It is a target rather than a bound: a leaf larger than the target is indivisible.

### Descriptor and geometry addressing

The format requires a positive integer `metadata.descriptor_block_rows` Q. The
storage row table is a permutation of `0..N`: entry i gives the physical vector row for
original feature ID i. Block b stores rows `[b*Q, min((b+1)*Q,N))`, with R equal
to that interval's length. Blocks are dense from zero through `ceil(N/Q)-1`,
using decimal names without leading zeroes. N = 0 has an empty storage-row entry
and no descriptor blocks. Vectors have the same scalar type and D as the forest.
Each vector occurs once in this table; different IDs with identical vector bytes
remain distinct rows. This is deduplication across trees, not across identities.

The writer defaults to a 2 KiB decoded descriptor-block target, giving Q = 16
for 128-byte uint8 SIFT descriptors. The forest builder defaults to 16-feature
leaves. These are tuning choices, not format constraints: readers use Q from
metadata and accept files built with other positive block targets or leaf sizes.
The [query measurements](../core/features/lazy-kdforest-query.md#current-access-path-and-performance-diagnosis)
explain the default selection.

The row permutation is explicit, not inferred from tree topology. A writer may
choose any order; readers use the stored map. Leaf membership/order and origin
mapping do not change when storage rows are reordered. Readers locate a vector by
division/remainder of its storage row by Q. A full verifier checks the map is a
permutation and validates all referenced vectors. Descriptor block b decodes to
exactly `R*D*scalar width` bytes. In SIFT mode, geometry block b decodes to
exactly `R*3*2*4` bytes and covers the same rows.

Every block is a complete independent zstd frame, and all C = `ceil(N/Q)` of them
are stored back to back in the single `features/corpus` entry, in block order.
`features/block_offsets` holds C+1 `uint64` values: block b occupies the stored
bytes `[offsets[b], offsets[b+1])` of that entry, measured from its first stored
byte. `offsets[0]` is zero, the values never decrease, and `offsets[C]` equals the
entry's stored length — a reader checks all three at open, because a truncated or
reordered offsets array would otherwise surface as an unexplained decode failure
on whichever block happened to be read first. Reading a block is therefore a seek
to a recorded offset followed by decoding one frame; nothing outside that range is
read or decoded. The optional geometry corpus has exactly C frames and follows
the same rules with its own offsets entry because its compressed frames have
different lengths. N = 0
has an empty storage-row entry, empty corpus entries, and each offsets entry
holds the single value zero.

### Hash composition

Every value in `content_hash.json.zst` is one digest, and there is exactly one of
them per section. The entry therefore has the same handful of hundred bytes for
three features and for a hundred million, and a reader budgets it as a constant.

**The digest function.** A digest is the 128-bit variant of XXH3 with seed 0 and
no secret, applied to a byte string. Its result is an unsigned 128-bit integer.
Where a digest is stored, it is written as that integer's 32 lowercase
hexadecimal digits, most significant digit first, zero-padded on the left.
Where a digest is itself hashed, it is written as that integer's 16 bytes, most
significant byte first, which is the byte string the 32 hexadecimal digits spell.

**The tree.** The digests form a Merkle tree of fixed depth, three levels at its
deepest:

1. A **leaf digest** is the digest of one item's decoded bytes. An item is one
   descriptor block, one geometry block, one origin block or one tree chunk. The
   table below says which bytes each kind of item contributes. Leaf digests are
   not stored in the file.
2. A **section digest** is either the digest of the section's decoded bytes,
   for a section the table defines as one byte string (`metadata`, `images`,
   `storage_rows`), or the *fold* of the section's leaf digests, for a section
   made of items (`origins`, `descriptors`, `geometry`, `trees`). A folded
   section node has one child per item, however many items there are; there are
   no intermediate nodes and no fixed fan-out.
3. The **content digest** `content_xxh128` is the fold of the section digests
   present in the file.

**Folding** a sequence of digests `d_0, d_1, ..., d_(n-1)` means taking the
digest of the byte string `B(d_0) || B(d_1) || ... || B(d_(n-1))`, where `B(d)`
is the 16-byte form above and `||` is concatenation. The sequence order is part
of the definition and is given per field in the table. Nothing else enters the
byte string: no count, no separator, no field name. Folding an empty sequence is
the digest of the empty byte string. Folding a sequence of one digest `d` is the
digest of `B(d)`, which is not `d`.

| Field | Digest of |
|-------|-----------|
| `metadata_xxh128` | The exact decoded bytes of `metadata.json.zst` |
| `images_xxh128` | The four `images/` entries' decoded bytes, concatenated in lexicographic path order |
| `origins_xxh128` | The fold of the origin blocks' digests, in numeric block order; a block digests its `image_indexes` decoded bytes followed by its `image_feature_indexes` decoded bytes |
| `storage_rows_xxh128` | The decoded row map |
| `descriptors_xxh128` | The fold of the descriptor blocks' digests, in numeric block order; a block digests its decoded vector bytes |
| `geometry_xxh128` | The fold of the geometry blocks' digests, in numeric block order; a block digests its decoded `3x2` float32 bytes |
| `trees_xxh128` | The fold of the chunk digests, in numeric tree order and then numeric chunk order within a tree; a chunk digests its `chunk` entry's decoded bytes |
| `content_xxh128` | The fold of the section digests above, in the order this table lists them, skipping those a generic file omits |

`images_xxh128`, `origins_xxh128` and `geometry_xxh128` are present exactly in
SIFT mode and absent in generic mode. `content_xxh128` covers only the other
fields; the hash entry itself is excluded.

A section made of many items is folded rather than hashed as one stream because
the leaf digests, which between them read every byte of the section, have no
dependency on one another: they can be computed independently, concurrently and
in any order. Only the fold is sequential, and its input is 16 bytes per item. A
writer or verifier that spreads the items across a thread pool therefore arrives
at the same section digest as one that walks them in a loop, and the result does
not depend on how many threads ran or how the items were batched.

Offset entries carry no digest. They are addressing rather than content, and a
wrong value is caught by the three structural checks on the offsets plus the
section digest over whatever they addressed.

This identity depends on packing, node layout and JSON serialization, but not
compression level. It is not a canonical identity of the vector set. Repacking
requires recomputing hashes. Hashes detect corruption, not malicious tampering.
ZIP CRC also covers each entry's stored compressed bytes.

### What is checked, and when

Opening validates the metadata, the integrity directory's own composition, the
expected entry set, lengths and references' declared ranges without reading any
chunk, and it hashes the two things it reads in full: `metadata.json.zst` and the
storage row map.

**Nothing else is hashed until a full verification asks for it.** Decoding a
chunk or a block checks its exact size and the local constraints its contents
must satisfy, and stops there; the digest that would confirm those bytes lives in
a section digest, which can only be computed by reading the whole section. A
query reads a few blocks out of millions, so it can afford neither. What this
buys is that a block read is a seek, a decode and a bounds check, at a cost that
does not depend on how large the file is.

Full verification is the one place a file's integrity is established. It
recomputes every section digest from the bytes on disk, folds them, and compares
both the sections and the whole-file digest against the directory; it also checks
reachability, ID permutations, split constraints, and that every tree describes
the same vectors. It necessarily reads the whole file. The digest check runs
first, then the structural checks.

All size arithmetic is checked before allocation, and readers may reject files
exceeding explicit resource limits. The ZIP dependency indexes the central
directory by filename, so opening separately counts its raw file records and
rejects a count greater than the unique-name index; otherwise a duplicate name
would be collapsed before entry-set validation could see it.

### Version

`metadata.json`'s `version` is 3. A reader accepts that number and no other, in
either direction: there is one on-disk shape at a time and no translation between
shapes. A file carrying another version is refused with a message naming the
version it holds, the version the reader wants, and the remedy — a `.kdf` is
derived from the `.sift` files it was built over, so rebuilding it is always
available and is always the fix.

Version 2 replaced version 1's two layouts (a shared descriptor corpus, or
descriptors copied into each tree) with the single shared corpus, and added the
SIFT-mode geometry corpus (`features/geometry` and its block offsets). Version 3
keeps that layout and changes only the integrity directory:
`content_hash.json.zst` holds one digest per section instead of one per descriptor
block, geometry block, origin block and tree chunk.

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
tens or hundreds of thousands of entries. At that scale the conventions have a
measurable cost:

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

**Vectors never join tree chunks.** They live once in the corpus because every
tree indexes the same feature identities. Keeping descriptor bytes out of tree
frames also avoids mixing their roughly 76% compression ratio with the much more
compressible topology columns.

**Each corpus is one entry of many frames.** Descriptor and optional geometry
blocks must stay
independently decodable — reading one descriptor may not require decompressing the
corpus — so they remain one frame each, and each is digested on its own before
those digests are folded into the section's. What changes is that the frames are
concatenated into a single entry and addressed by a stored offsets array instead
of by the ZIP directory.

This is the departure that matters most, because the useful configurations use
small blocks: at 16 KiB on DinoLedge, descriptor blocks were ~76,000 of ~100,000
entries and become one corpus entry plus one offsets entry, taking the measured
version-1 open latency from 674 ms to 130 ms. SIFT geometry adds its own two
entries. It also makes block size
free in entry count, which it was not before — the choice is now purely about
read granularity. Unlike the grouping above it costs nothing in compression,
because the frames are unchanged; only their addressing moved.

The entries are named `.frames` rather than `.zst` so that a tool which assumes
one frame per entry reports an error instead of decoding only the first block and
reporting success. Descriptor and geometry blocks use the same logical row
boundaries but separate offsets because their compressed lengths differ.

The properties a reader relies on are preserved. The file is still a ZIP of
STORE entries, so standard tools still list it and still extract any entry. Binary
arrays are still little-endian and row-major with no headers of their own. Names
still encode shape and type, and every decoded length is still derivable from a
name and checked against it. The digests still cover decoded bytes and still
identify the same byte sequences they would have identified had those bytes sat
in separate entries. A reader that knows this section can
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
dependency on `sfmtool-core`; a write reports how far along it is, and is
cancelled, through the `sfmtool-progress` parameter both crates share, which
depends on nothing but `std` for exactly this reason. One write entry point
takes that parameter, rather than a silent one and a reporting one, and a caller
with nothing to report through passes the progress that reports nothing. A write
refuses a destination that already holds a file unless it is told that one may
be replaced, and either way the archive is streamed into a temporary sibling in
the destination's own directory and renamed over the destination once it is
whole, so a write that fails or is cancelled leaves what was there untouched and
nothing of its own beside it. Core owns forest construction and queries, in
[`features/kdforest/persistent.rs`](../../crates/sfmtool-core/src/features/kdforest/persistent.rs).
The `uint8` half of both is bound for Python on the `sfmtool.spatial`
submodule, in
[`spatial/kdf.rs`](../../crates/sfmtool-py/src/spatial/kdf.rs). `float32` is not bound: the
eager `KdForest` it would be compared against is `uint8` only.

Full verification is
[`verify_kdf`](../../crates/sfmtool-kdf-format/src/verify.rs), which first calls
[`KdfFile::verify_content`](../../crates/sfmtool-kdf-format/src/read.rs) for the
digest check and then runs the structural checks.

The writer streams descriptor and geometry frames into their ZIP entries and
retains the offsets and hashes; it does not buffer the complete compressed corpus.
The summary derives array sizes from entry shapes, including the uint64 block
offsets. JSON decompression is bounded by `max_metadata_bytes`; the ZIP STORE size
alone does not bound the expanded JSON. Array-shape arithmetic checks overflow.

The measurements that chose this layout — the size of a 9.7M-descriptor file
under the rejected tree-local layout and under the shared corpus, and why the
corpus is compressed in blocks rather than stored as a flat array — are recorded
in [KDF layout measurements](../core/features/kdf-layout-measurements.md).
