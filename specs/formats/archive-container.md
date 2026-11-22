# The archive container

The archive container is a ZIP file whose entries are all stored uncompressed
at the ZIP level, each entry holding
[zstandard](https://en.wikipedia.org/wiki/Zstd)-compressed compact UTF-8 JSON
or raw little-endian numeric arrays, with XXH128 digests of the
*uncompressed* bytes kept in a `content_hash.json.zst` entry. The `.sift`,
`.matches`, `.sfmr`, `.camrig` and `.kdf` formats are all built on it. The
digests let a reader tell whether a file is intact, and let other files refer to
it by content rather than by path. This document specifies what the bytes on disk
look like and how the digests are composed.

It describes no format's contents. Each of the five formats defines its own entry
list, schemas and validation rules in its own spec:
[sift-file-format.md](sift-file-format.md),
[matches-file-format.md](matches-file-format.md),
[sfmr-file-format.md](sfmr-file-format.md),
[camrig-file-format.md](camrig-file-format.md),
[kdf-file-format.md](kdf-file-format.md). The container is the part they share,
so that the five do not drift into five different files that only look alike.

## The container on disk

### ZIP, with no ZIP-level compression

A container file is a ZIP archive whose entries are all written with the STORE
method. ZIP supplies the directory — any entry can be located and read without
touching the rest — and compression is left entirely to the per-entry layer
below, so nothing is compressed twice. Standard tools (`unzip`, any language's
zip library) open the file, which is what makes the "pull it apart from the shell"
recipes in the format specs work.

### What the entry rules are for

The rules below say what an entry holds, how it is compressed and how it is
named. They exist to give four things:

- **A reader decodes only the data it needs.** It reads the entries it wants
  and skips the rest.
- **Opening a file is cheap.** A reader parses the whole ZIP central directory
  before it can read any entry, and each entry costs about 164 bytes of local
  header and central directory record and about 5.6 µs of open time, as
  measured on `.kdf` files
  ([kdf-file-format.md § How this format meets the container rules](kdf-file-format.md#how-this-format-meets-the-container-rules)).
  The open cost grows with the entry count.
- **Every decoded length is known before decompressing.** The entry name gives
  it, and a reader checks the decoded bytes against it.
- **A file can be taken apart with standard tools.** Someone with only the
  file, `unzip` and `zstd` can extract every entry and decode it.

### What an entry holds

A format's entries are of two kinds.

**An entry the schema fixes** is one whose count does not change with the
size of the data. Each numeric field a consumer may read on its own is its own
entry, one primitive type per entry, so a consumer reads the columns it needs,
and similar values compress together. Small descriptive values that are read
together at open, such as a format's metadata, sit in a JSON entry. Every entry
of `.sift`, `.matches`, `.sfmr` and `.camrig` is of this kind, as are `.kdf`'s
metadata, image table, storage-row map and offsets entries.

**An entry repeated per chunk or block of the data** makes the format's entry
count grow with its data. Arrays that every consumer reads together may share
one such entry, because a separate entry for each would multiply the entry
count and the open cost with it. An entry that groups arrays:

- names every count its arrays' shapes use, so its decoded length is known
  before it is decompressed and is checked against it;
- has its layout, each array's name, element type and shape in the order they
  are stored, described in the format's metadata;
- is recorded in the format spec with the measured open-time saving set
  against the compression cost of sharing one zstd frame.

Bulk data and the index arrays that address it do not share an entry, even
when they are read together, because each compresses worse in a shared zstd
frame. `.kdf` tried folding its descriptor vectors into the entry that holds a
tree chunk's integer arrays and reverted it: one zstd frame holding both the
descriptors, which compress to about 76% of their size, and the integer arrays,
which compress to about 46%, compressed each worse than two frames did.

`.kdf` is the format whose entry count grows with its data: it has an entry
per tree chunk and two per origin block. Its tree chunk entry groups three
arrays, and its origin entries hold one field each.
[kdf-file-format.md § How this format meets the container rules](kdf-file-format.md#how-this-format-meets-the-container-rules)
records the measurements. `.kdf`'s `metadata.json` describes the chunk
entry's layout in a `chunk_layout` field
([kdf-file-format.md § Metadata and versioning](kdf-file-format.md#metadata-and-versioning)).

### An entry is zstandard frames, and its suffix says how many

An entry's stored bytes are zstd frames, and the last part of its name says how
many:

- **`.zst`**: exactly one frame, decompressed as a whole. Every entry of
  `.sift`, `.matches`, `.sfmr` and `.camrig` is one, as is every `.kdf` entry
  but the two below.
- **`.frames`**: independently decodable frames, one per block of rows, stored
  back to back in block order. A companion `.zst` entry of offsets gives each
  frame's byte range within the entry, so a reader decodes one block by seeking
  to its offset and decoding one frame, and decodes nothing else. Each block's
  decoded bytes are digested on their own, and the block digests are folded
  into the section's digest, in an order and by a rule the format fixes.

The `.frames` form is for data that must be readable block by block, where an
entry per block would make the entry count grow with the data. `.kdf`'s
descriptor corpus and its optional geometry corpus are the two `.frames`
entries.

A stock `zstd -d` decodes every frame of a concatenation in order, so `unzip -p`
piped into `zstd -d` recovers a `.frames` entry's whole decoded array, as it
does a `.zst` entry's; only reading one block on its own needs the offsets.
Some decoders stop after the first frame instead: Python's
`compression.zstd.ZstdDecompressor` returns the first frame's bytes and keeps
the rest as unused data. The two suffixes differ so that a tool that decodes
each `.zst` entry with such a decoder does not take a `.frames` entry for one
and return its first block as the whole entry.

zstd is chosen over the ZIP-native deflate for the ratio and for decompression
speed on the large numeric columns. The compression level is the writer's
choice and is not recorded in the file; a reader accepts a frame written at any
level.

Two kinds of payload sit under the frames:

- **JSON** — a writer stores compact UTF-8 JSON: no whitespace between tokens
  and no trailing newline. The digests are taken over these stored bytes, so the
  same values spelled differently give a different digest, and a consumer that
  rewrites a JSON entry in any other spelling must recompute every digest that
  covers it. A reader must accept any valid JSON and must not require a
  particular formatting.
- **Raw binary** — numeric arrays in C (row-major) order, little-endian, with no
  header of their own; their shape and element type come from the entry name,
  and for an entry that groups arrays, from the layout its format describes.

Numeric data is always little-endian, whatever the machine that wrote it. A
writer or reader on a big-endian host byte-swaps each element.

### Entry names encode shape and type

A binary entry is named `{field_name}.{dim1}.{dim2}….{dtype}.zst`, so a reader
knows exactly how many bytes to expect before it decompresses anything:
`positions_xyzw.2107.4.float64.zst` is 2107 × 4 `float64` values and must
decompress to exactly `2107 * 4 * 8 = 67424` bytes. A reader checks that byte
count and rejects the entry if it disagrees, which is the format's shape check.
A `.frames` entry is named the same way with `.frames` in place of `.zst`, and
its name gives the shape of the whole decoded array. An entry that groups
arrays names every count its arrays' shapes use, as `.kdf`'s
`chunk.{M}.{P}.{scalar_type}.zst` does, and its decoded length is the sum of
its arrays' lengths.
Element types are spelled as `uint8`/`uint16`/`uint32`/`uint64`,
`int8`/`int16`/`int32`/`int64`, `float16`/`float32`/`float64`, and `uint128` for a raw
16-byte XXH128 digest column; which of them a format uses is the format's
business.

### A file explains itself

Someone with a file, `unzip` and `zstd`, but not its spec, can find every
array's shape, element type and meaning: for an entry that holds one field,
from its name; for an entry that groups arrays, from the layout description in
the format's metadata. An array of codes has its legend, the name each code
stands for, in the metadata, as `.sfmr`'s `point_constraints` column has
`point_constraint_names` and `.kdf`'s node `kind` column has `node_kinds`.
Names, whether of an entry, a JSON field or a column, are chosen to say what
they hold.

### Content hashes

Every container format stores its integrity hashes in a `content_hash.json.zst`
entry, whose fields are 32-character lowercase hexadecimal XXH128 digests. XXH128
is not cryptographic; it is chosen for throughput (GB/s) with collision
resistance good enough that a digest can be used as an identity — a `.sfmr` point
ID and the `.sift` links inside a `.matches` file both lean on that.

Three rules hold across all five formats:

1. **Hashes are taken over uncompressed bytes.** A verifier decompresses an entry
   and hashes the bytes it got, never re-serialized JSON — re-serializing would
   make the digest depend on the writer's float formatting and JSON library. A
   consequence worth stating: the digests are independent of the zstd level, so
   the same data rewritten at a different level keeps its identity.
2. **A section digest is XXH128 over the concatenated uncompressed bytes of that
   section's entries, in an order the format fixes.** Entries are fed into one
   streaming hasher, so the digest sees exactly the bytes of the entries and
   nothing separating them; the order is part of the format's contract (`.sfmr`
   and `.matches` group entries into sections and hash each section's entries in
   lexicographic path order, `.kdf` fixes a numeric tree/chunk and block order,
   while `.sift` and `.camrig` make each hashed entry its own one-entry section). An optional entry participates only when it is present,
   which is why each format spec spells out what is in each of its sections under
   which conditions.
3. **The whole-file digest is XXH128 over the concatenated section digests, each
   written as 16 bytes big-endian**, in the order the format lists, skipping
   absent optional sections. This one field is called `content_xxh128` in all five
   formats. A format may also declare a present section **derived** and leave it
   out of this digest: its bytes are still covered by their own section hash and
   still verified, but they are recomputable from sections that are in the digest
   and so say nothing about which value the file holds. `.sfmr` does this with
   its `derived/` section; no other format has one.

Note the two byte orders, which are deliberately different and easy to confuse:
numeric *data* is little-endian, while a 128-bit *digest* being folded into
another hash is serialized big-endian (most significant byte first).

`content_hash.json.zst` is the only entry no hash covers — it is where the hashes
land. Formats add their own fields beside `content_xxh128` (a per-section digest,
a `metadata_xxh128`, `.sift`'s `feature_tool_xxh128`); those field lists live in
the format specs.

Verification belongs to each format, not to the container: a verifier recomputes
the digests above in the order its format fixes, and also checks the structural
constraints only that format defines.

## Implementations

In this repository the container is implemented once, in the
[`sfmtool-archive-io`](../../crates/sfmtool-archive-io/) crate, and each format
crate builds its reader, writer and verifier on it. That crate's interface and
design are specified in [archive-io-crate.md](archive-io-crate.md).

## Non-goals

- **No schema.** The container defines no entry names beyond
  `content_hash.json.zst`, no required entries, no versions and no metadata
  fields; each format defines those.
- **No verification rules of its own.** The container fixes how a digest is
  composed; which sections exist, their order, and what else makes a file valid
  are each format's.
- **No ZIP-level compression, encryption, or multi-file spanning.** Entries are
  always STORE, and readers of these files rely on it.
