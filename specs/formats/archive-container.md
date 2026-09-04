# The archive container

Every file sfmtool writes for its own consumption — extracted features, feature
matches, reconstructions, camera rigs — is the same kind of file underneath: a
ZIP archive that applies no compression of its own, holding entries that are
each compressed with [zstandard](https://en.wikipedia.org/wiki/Zstd), where
metadata is compact JSON and bulk numeric data is raw little-endian binary, one
entry per column. Written alongside them are XXH128 hashes taken over the
*uncompressed* bytes, so a reader can tell whether the file is intact and other
files can refer to it by content rather than by path. That shared shape is the
**archive container**, and this document is its specification: what the bytes on
disk look like, how the hashes are composed, and the Rust primitives the format
crates call to read and write them.

It describes no format's contents. `.sift`, `.matches`, `.sfmr` and `.camrig`
each own their entry list, their schemas, their validation rules and their error
type, and each has its own spec:
[sift-file-format.md](sift-file-format.md),
[matches-file-format.md](matches-file-format.md),
[sfmr-file-format.md](sfmr-file-format.md),
[camrig-file-format.md](camrig-file-format.md). The container is what stops those
four from drifting into four different files that merely look alike.

The on-disk contract comes before the Rust interface here, because that is what
the four format specs link to this one for; a caller who wants the functions can
skip to [Rust API](#rust-api).

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
name ends in `.zst`. zstd is chosen over the ZIP-native deflate for the ratio and
for decompression speed on the large numeric columns. The level is a **choice of
each format's writer**, not a property of the container: `write_sift`,
`write_matches` and `write_camrig` take it as an argument and `.sfmr` carries it
on `WriteOptions`, whose default is 3. The Python bindings default to 3 as well,
except `.sift`'s, which defaults to 5.

Two kinds of payload sit under that frame:

- **JSON** — compact, no pretty-printing, no trailing newline: exactly what
  `serde_json::to_vec` produces. Compactness is not cosmetic; the metadata hashes
  are taken over these bytes, so any reformatting invalidates them.
- **Raw binary** — a numeric array in C (row-major) order, little-endian, with no
  header of its own; its shape and element type come from the entry name.

Little-endian is not negotiable: the binary columns are stored in the layout a
little-endian machine holds them in, and the crate refuses to compile for a
big-endian target rather than silently producing byte-swapped files.

Tables are columnar — one entry per field, one primitive type per entry — so a
consumer can read the columns it needs and skip the rest, and so similar values
compress together.

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

Three rules hold across all four formats:

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
   lexicographic path order, while `.sift` and `.camrig` make each hashed entry
   its own one-entry section). An optional entry participates only when it is present,
   which is why each format spec spells out what is in each of its sections under
   which conditions.
3. **The whole-file digest is XXH128 over the concatenated section digests, each
   written as 16 bytes big-endian**, in the order the format lists, skipping
   absent optional sections. This one field is called `content_xxh128` in all four
   formats.

Note the two byte orders, which are deliberately different and easy to confuse:
numeric *data* is little-endian, while a 128-bit *digest* being folded into
another hash is serialized big-endian (most significant byte first).

`content_hash.json.zst` is the only entry no hash covers — it is where the hashes
land. Formats add their own fields beside `content_xxh128` (a per-section digest,
a `metadata_xxh128`, `.sift`'s `feature_tool_xxh128`); those field lists live in
the format specs.

Verification is per-format code, not container code: each format crate has a
`verify_*` function that recomputes the digests above and also checks the
structural constraints only it knows about.

## Rust API

The primitives live in
[sfmtool-archive-io/src/lib.rs](../../crates/sfmtool-archive-io/src/lib.rs) and are
used by the four format crates
([sift-format](../../crates/sift-format/),
[matches-format](../../crates/matches-format/),
[sfmr-format](../../crates/sfmr-format/),
[camrig-format](../../crates/camrig-format/)) and by nothing else. There are no
Python bindings: Python reaches these bytes through each format's own binding.

```rust
/// Errors from archive I/O. Each format crate converts this into its own
/// public error type, so callers of `read_sfmr` / `write_matches` / … never
/// see it.
pub enum ArchiveIoError { Io(..), Zip(..), Json(..), InvalidFormat(String), ShapeMismatch(String) }

// Reading
pub fn read_zst_entry<R: Read + Seek>(archive: &mut ZipArchive<R>, name: &str)
    -> Result<Vec<u8>, ArchiveIoError>;
pub fn read_json_entry<R: Read + Seek, T: DeserializeOwned>(archive: &mut ZipArchive<R>, name: &str)
    -> Result<T, ArchiveIoError>;
pub fn read_binary_array<R: Read + Seek, T: bytemuck::Pod>(
    archive: &mut ZipArchive<R>, name: &str, expected_len: usize)
    -> Result<Vec<T>, ArchiveIoError>;
pub fn read_uint128_array<R: Read + Seek>(archive: &mut ZipArchive<R>, name: &str, count: usize)
    -> Result<Vec<[u8; 16]>, ArchiveIoError>;

// Raw buffer reinterpretation, for verifiers working from bytes they hashed
pub fn raw_to_u32(raw: &[u8]) -> Cow<'_, [u32]>;
pub fn raw_to_f32(raw: &[u8]) -> Cow<'_, [f32]>;
pub fn raw_to_f64(raw: &[u8]) -> Cow<'_, [f64]>;

// Writing
pub fn zstd_compress(data: &[u8], level: i32) -> Result<Vec<u8>, ArchiveIoError>;
pub fn write_json_entry<W: Write + Seek>(
    zip: &mut ZipWriter<W>, name: &str, value: &impl Serialize, zstd_level: i32)
    -> Result<Vec<u8>, ArchiveIoError>;                       // returns the uncompressed JSON
pub fn write_binary_entry<W: Write + Seek>(
    zip: &mut ZipWriter<W>, name: &str, data: &[u8], zstd_level: i32)
    -> Result<(), ArchiveIoError>;
pub fn write_binary_entry_hashed<W: Write + Seek>(
    zip: &mut ZipWriter<W>, name: &str, data: &[u8], zstd_level: i32, hasher: &mut Xxh3)
    -> Result<(), ArchiveIoError>;

// Hashing
pub fn format_hash(digest: u128) -> String;                   // 32-char lowercase hex
```

**Why this shape.** The surface is entry-at-a-time rather than a
"Container" object with an entry table, because the four formats disagree about
almost everything above the entry: which entries exist, whether one is optional,
what a section is, what the metadata means. What they genuinely share is one
entry's worth of work — compress it, store it, fold it into a hash — so that is
what the crate owns. Pushing the entry table up here would mean a schema
description language, and every format crate would then be a client of it rather
than of a few functions.

Three consequences of that choice are visible in the signatures:

- **The caller owns the hasher.** `write_binary_entry_hashed` takes
  `&mut Xxh3` and updates it with the uncompressed bytes; it does not decide what
  a section is or when a digest is finished. The alternative — a section object
  that opened and closed sections — would have to model optional sections and
  per-format ordering, which is exactly the part that differs.
- **Write and hash are one call.** They are separable (`write_binary_entry` plus
  a manual `hasher.update`) and the JSON path is separate by necessity, but the
  binary path pairs them so an entry cannot be written and left out of the hash,
  or hashed in an order that does not match the write order.
- **JSON writing returns bytes, binary writing does not.**
  `write_json_entry` serializes fresh bytes the caller has no other handle on and
  hands them back for hashing; a binary entry's bytes are the caller's own buffer
  — a whole column of positions, thumbnails or patch bitmaps — so returning a
  copy would double the peak memory of a large write for nothing.

Errors are one enum rather than `Result<_, String>` so each format crate can map
variant to variant into its own public error (`SfmrError`, `MatchesError`,
`SiftError`, `CamRigError`) and keep `ArchiveIoError` out of its public API.

**Example — write two hashed columns, then read one back:**

```rust
use sfmtool_archive_io::{read_binary_array, write_binary_entry_hashed, format_hash};
use xxhash_rust::xxh3::Xxh3;
use zip::{ZipArchive, ZipWriter};

let positions: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0];
let indexes: Vec<u32> = vec![0, 1];

let mut zip = ZipWriter::new(std::io::Cursor::new(Vec::new()));
let mut section = Xxh3::new();
write_binary_entry_hashed(&mut zip, "points/indexes.2.uint32.zst",
                          bytemuck::cast_slice(&indexes), 3, &mut section)?;
write_binary_entry_hashed(&mut zip, "points/positions_xy.2.2.float64.zst",
                          bytemuck::cast_slice(&positions), 3, &mut section)?;
let section_digest = section.digest128();          // → content_hash.json.zst
let hex = format_hash(section_digest);

let buf = zip.finish()?.into_inner();
let mut archive = ZipArchive::new(std::io::Cursor::new(buf))?;
let read: Vec<f64> = read_binary_array(&mut archive, "points/positions_xy.2.2.float64.zst", 4)?;
```

Note the entry names: the container treats them as opaque strings, so the
`{field}.{dims…}.{dtype}.zst` convention is enforced by each format crate
building its own names (`sfmr-format`'s and `matches-format`'s `entries` modules
do this from the counts in the metadata), and by the reader passing the element
count it expects.

## Implementation notes

**Alignment is the hazard the read path is built around.** A `Vec<u8>` returned
by the decompressor promises only 1-byte alignment, while reinterpreting it as
`u32`/`f32`/`f64` needs 4 or 8. `bytemuck::cast_slice` *panics* on a misaligned
buffer, so both read paths — `read_binary_array` (via its private `cast_or_copy`
helper) and the `raw_to_*` family — try the cast first and fall back to copying
through a freshly allocated, correctly aligned `Vec<T>`. The fast path is the
common one, because allocators over-align sizeable allocations; the fallback
exists so that "common" never becomes "load-bearing". Which branch runs is not
observable in the result.

**`read_binary_array` and `raw_to_*` disagree about trailing bytes on purpose.**
`read_binary_array` knows the expected element count and rejects any length
mismatch with a `ShapeMismatch` naming the entry — that is the format's shape
check. `raw_to_*` is used by verifiers that already hold the bytes they hashed
and are re-reading them for structural checks; it silently drops a trailing
partial element rather than panicking, on the grounds that a truncated entry is
about to fail a structural check anyway and the verifier's job is to report, not
to abort.

**The misaligned branch is not reachable through the archive.** Whether a
decompressed buffer lands aligned is the allocator's choice, which is why
`cast_or_copy` is split out of `read_binary_array` at all: the test constructs a
guaranteed-misaligned slice and calls the helper directly, so folding it back
into its caller would leave the fallback untested.

**Neither the hasher nor the compressor is re-exported.** Format crates depend
on `xxhash-rust` themselves, because `write_binary_entry_hashed` takes an `Xxh3`
by reference and the section-level hashing lives in their write paths. A format crate that wants
`zstd` directly (only the round-trip tests do) declares it as a dev-dependency;
the library paths all go through `zstd_compress`.

## Testing

[sfmtool-archive-io/src/tests.rs](../../crates/sfmtool-archive-io/src/tests.rs)
covers the primitives in isolation: JSON and binary round trips (including the
empty array), the element-count rejections for `read_binary_array` and
`read_uint128_array`, `format_hash`'s zero-padded lowercase output, and the three
failure modes a corrupt file produces — a missing entry is a `Zip` error, a
non-zstd payload an `InvalidFormat` error naming the entry, and a well-formed
entry whose payload is not JSON a `Json` error. One test pins that
`zstd_compress` really honours its level, on a payload chosen so the level can
show: a repetitive buffer bottoms out at the same size at every level and
incompressible noise comes back byte-identical, so it uses pseudo-random data
over a four-symbol alphabet, where the extra search effort pays.

Two properties get dedicated tests because everything above rests on them:
`write_binary_entry_hashed` must produce the same digest as hashing the two
buffers by hand in write order *and* leave both entries readable in the archive,
and the alignment fallbacks must return the same values as the borrowing path for
a slice deliberately placed at an address congruent to 1 modulo the element
alignment (derived from the real base address, not by prepending a byte and
hoping). Whole-file round trips through real data live in each format crate's own
tests.

## Non-goals

- **No schema.** The container does not know entry names, required entries,
  versions or metadata; a format crate supplies all of that.
- **No verification.** Recomputing and comparing hashes is each format's
  `verify_*` function, since only the format knows its sections and its
  structural constraints.
- **No streaming or partial writes.** Entries are compressed in one shot from a
  buffer the caller already holds, and a file is written whole.
- **No big-endian targets.** The crate fails to compile rather than write
  byte-swapped columns; a big-endian reader would have to swap on read, which
  nothing implements.
- **No ZIP-level compression, encryption, or multi-file spanning.** Entries are
  always STORE, and readers of these files rely on it.
