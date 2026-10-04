# The archive container

The archive container is a ZIP file whose entries are all stored uncompressed
at the ZIP level, each entry normally holding one
[zstandard](https://en.wikipedia.org/wiki/Zstd) frame of either compact UTF-8
JSON or a raw little-endian numeric array, with XXH128 digests of the
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

### Every entry is a zstandard frame

Each entry's stored bytes are a single zstd frame, and by convention every entry
name ends in `.zst`. The one exception among the five formats is `.kdf`'s shared
descriptor corpus, which concatenates one frame per block into a single entry and
is named `.frames` to say so; `.kdf` explains why under
[Where this format departs from the container conventions](kdf-file-format.md#where-this-format-departs-from-the-container-conventions).
A reader of any other format may assume one frame per entry. zstd is chosen over the ZIP-native deflate for the ratio and
for decompression speed on the large numeric columns. The compression level is
the writer's choice and is not recorded in the file; a reader accepts a frame
written at any level.

Two kinds of payload sit under that frame:

- **JSON** — a writer stores compact UTF-8 JSON: no whitespace between tokens
  and no trailing newline. The digests are taken over these stored bytes, so the
  same values spelled differently give a different digest, and a consumer that
  rewrites a JSON entry in any other spelling must recompute every digest that
  covers it. A reader must accept any valid JSON and must not require a
  particular formatting.
- **Raw binary** — a numeric array in C (row-major) order, little-endian, with no
  header of its own; its shape and element type come from the entry name.

Numeric data is always little-endian, whatever the machine that wrote it. A
writer or reader on a big-endian host byte-swaps each element.

Tables are columnar — one entry per field, one primitive type per entry — so a
consumer can read the columns it needs and skip the rest, and so similar values
compress together.

This convention assumes what is true of every format here but one: that the entry
count follows the *schema*, so it stays small however large the data. `.kdf` is the
exception — it has an entry per tree chunk and per descriptor block, so entry count
grows with the corpus — and it groups a chunk's integer arrays into one entry for
that reason.

Two conditions made that sound there, and both are the test to apply before doing
the same elsewhere. The fields must be **always read together**, so no consumer
ever wants one without the others. And they must **compress alike**: `.kdf` tried
folding its descriptor vectors in beside those integer columns and reverted it,
because one zstd frame holding both 76%-compressible bulk data and 46%-compressible
columns compressed each worse than two frames did. Where a grouping is made, the
format spec says what lies at which offset.

### Entry names encode shape and type

A binary entry is named `{field_name}.{dim1}.{dim2}….{dtype}.zst`, so a reader
knows exactly how many bytes to expect before it decompresses anything:
`positions_xyzw.2107.4.float64.zst` is 2107 × 4 `float64` values and must
decompress to exactly `2107 * 4 * 8 = 67424` bytes. A reader checks that byte
count and rejects the entry if it disagrees, which is the format's shape check.
Element types are spelled as `uint8`/`uint16`/`uint32`/`uint64`,
`int8`/`int16`/`int32`/`int64`, `float32`/`float64`, and `uint128` for a raw
16-byte XXH128 digest column; which of them a format uses is the format's
business.

Names, whether of an entry, a JSON field or a column, are chosen to be
self-documenting: someone who opens one of these files without having read its
spec should be able to work out what they are looking at.

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
   its `derived/` section from version 10; no other format has one.

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
