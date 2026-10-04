# The archive-io crate

`sfmtool-archive-io` is the Rust crate that reads and writes the
[archive container](archive-container.md) — the ZIP file of zstd-compressed JSON
and little-endian numeric entries, with XXH128 digests of the uncompressed bytes,
that `.sift`, `.matches`, `.sfmr`, `.camrig` and `.kdf` are built on. It provides
one entry's worth of work at a time: compress and store an entry, read one back
with its length checked, fold uncompressed bytes into a digest, and write a whole
file without leaving a partial one at its target. The format crates call it; it
knows none of their entry names, sections or rules. The on-disk contract it
implements is in [archive-container.md](archive-container.md); this document
covers the Rust interface and why it is shaped that way.

## Rust API

The primitives live in
[sfmtool-archive-io/src/lib.rs](../../crates/sfmtool-archive-io/src/lib.rs). The
five format crates
([sfmtool-sift-format](../../crates/sfmtool-sift-format/),
[sfmtool-matches-format](../../crates/sfmtool-matches-format/),
[sfmtool-sfmr-format](../../crates/sfmtool-sfmr-format/),
[sfmtool-camrig-format](../../crates/sfmtool-camrig-format/),
[sfmtool-kdf-format](../../crates/sfmtool-kdf-format/)) build their readers,
writers and verifiers on them. The one other user is `sfmtool-core`, whose
in-memory reconstruction hash in
[reconstruction/edited.rs](../../crates/sfmtool-core/src/reconstruction/edited.rs)
calls `format_hash` so it spells a digest the way the files do. There are no
Python bindings: Python reaches these bytes through each format's own binding.

The zstd level is passed in by each format's writer: `write_sift`,
`write_matches` and `write_camrig` take it as an argument, `.sfmr` carries it on
`WriteOptions` and `.kdf` on `KdfWriteOptions`, both defaulting to 3. The Python
bindings default to 3 as well, except `.sift`'s, which defaults to 5.

```rust
/// Errors from archive I/O. Each format crate converts this into its own
/// public error type, so callers of `read_sfmr` / `write_matches` / … never
/// see it.
pub enum ArchiveIoError { Io(..), Zip(..), Json(..), InvalidFormat(String), ShapeMismatch(String) }

// The `workspace` object that `.sfmr` and `.matches` metadata embed;
// those two format crates re-export both types.
pub struct WorkspaceMetadata { pub absolute_path: String, pub relative_path: String,
                               pub contents: WorkspaceContents }
pub struct WorkspaceContents { pub feature_tool: String, pub feature_type: String,
                               pub feature_options: serde_json::Value,
                               pub feature_prefix_dir: String }

// Whole-file content identity: push section digests in format-defined order.
pub struct SectionDigests { /* streaming XXH128 hasher */ }
impl SectionDigests {
    pub fn new() -> Self;
    pub fn push(&mut self, digest: u128);
    pub fn push_bytes(&mut self, uncompressed_entry: &[u8]);
    pub fn finish(self) -> u128;
}

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

// Reading a whole archive: one sequential pass, then parallel decompression
pub struct DecodedEntries { /* … */ }
impl DecodedEntries {
    pub fn read_all<R: Read + Seek>(archive: &mut ZipArchive<R>) -> Result<Self, ArchiveIoError>;
    pub fn zst_entry(&self, name: &str) -> Result<&[u8], ArchiveIoError>;
    pub fn json_entry<T: DeserializeOwned>(&self, name: &str) -> Result<T, ArchiveIoError>;
    pub fn binary_array<T: bytemuck::Pod>(&self, name: &str, expected_len: usize)
        -> Result<Vec<T>, ArchiveIoError>;
    pub fn uint128_array(&self, name: &str, count: usize)
        -> Result<Vec<[u8; 16]>, ArchiveIoError>;
}

// Raw buffer reinterpretation, for verifiers working from bytes they hashed
pub fn raw_to_u32(raw: &[u8]) -> Cow<'_, [u32]>;
pub fn raw_to_f32(raw: &[u8]) -> Cow<'_, [f32]>;
pub fn raw_to_f64(raw: &[u8]) -> Cow<'_, [f64]>;

// Writing
pub fn zstd_compress(data: &[u8], level: i32) -> Result<Vec<u8>, ArchiveIoError>;
pub fn json_entry_bytes(value: &impl Serialize) -> Result<Vec<u8>, ArchiveIoError>;
                                                              // the bytes write_json_entry stores
pub fn write_json_entry<W: Write + Seek>(
    zip: &mut ZipWriter<W>, name: &str, value: &impl Serialize, zstd_level: i32)
    -> Result<Vec<u8>, ArchiveIoError>;                       // returns the uncompressed JSON
pub fn write_binary_entry<W: Write + Seek>(
    zip: &mut ZipWriter<W>, name: &str, data: &[u8], zstd_level: i32)
    -> Result<(), ArchiveIoError>;
pub fn write_binary_entry_hashed<W: Write + Seek>(
    zip: &mut ZipWriter<W>, name: &str, data: &[u8], zstd_level: i32, hasher: &mut Xxh3)
    -> Result<(), ArchiveIoError>;
pub fn write_atomically<T, E: From<std::io::Error>, F: FnOnce(&mut File) -> Result<T, E>>(
    path: &Path, write: F) -> Result<T, E>;

// Hashing
pub fn format_hash(digest: u128) -> String;                   // 32-char lowercase hex
```

**Why this shape.** The surface is entry-at-a-time rather than a
"Container" object with an entry table — even `DecodedEntries`, which holds a
whole file, is keyed by name and knows no schema — because the five formats disagree about
almost everything above the entry: which entries exist, whether one is optional,
what a section is, what the metadata means. What they genuinely share is one
entry's worth of work — compress it, store it, fold it into a hash — so that is
what the crate owns. Pushing the entry table up here would mean a schema
description language, and every format crate would then be a client of it rather
than of a few functions.

**A write never leaves a partial file at its target.** `write_atomically` is the
one piece of the writing surface that is not entry-at-a-time, and every format's
`write_*` entry point goes through it. It creates `<file name>.tmp<hex>` in the
**target's own directory**, hands the handle to the closure that streams the
whole archive, flushes and syncs it, and only then renames it over the target;
the temporary file is in that directory rather than a system temp one so the last
step is a rename on one filesystem rather than a copy across two, and `rename`
replaces an existing target on every platform this builds for. Any failure, a
panic unwinding out of the closure included, removes the temporary file and
leaves the target exactly as it was. A **missing parent directory is an error**,
as it is for a direct write: whether a path inside one should bring it into
existence is a decision about what that path means, and it stays with the format
crate, some of which create the parent first and some of which deliberately do
not. The reason it is here rather than in one
format crate is that the situation is every format's: a writer's target is often
the only copy of what it is replacing, and a plain `File::create` truncates that
copy at the moment it opens it, so a failure part-way through leaves a partial
archive where the original was. The in-memory hashing path, which serialises into
a buffer to compute a content hash without writing anything, does not go through
it and is unaffected.

`SectionDigests` handles only the final fold: `push` appends a section's digest
as 16 big-endian bytes, `push_bytes` first hashes an entry's uncompressed bytes
for one-entry sections, and `finish` returns XXH128 over that stream. The format
still decides which sections exist and their order. For example, a format with
metadata and an optional images section calls `push_bytes(&metadata_raw)`, then
`push(images_digest)` only when images are present, then `finish()`.

Three consequences of the entry-at-a-time choice are visible in the signatures:

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

### Reading a whole archive

A read that consumes most of a file goes through `DecodedEntries` instead of
asking for entries one at a time. `read_all` walks the ZIP once, collecting each
entry's stored bytes in archive order, and then expands the zstd frames — which
are independent of one another — in parallel; the typed accessors serve the
resulting buffers by name. They mirror `read_json_entry`, `read_binary_array`
and `read_uint128_array` and share their conversion and length checks, so the
two paths return the same values and the same errors, down to the entry name in
the message and the `Zip` `FileNotFound` for an entry the archive does not hold.
An optional entry is therefore still just an entry a format crate does not ask
for.

This is a separate entry point rather than a change of behaviour inside
`read_zst_entry` so that the parallel region is something a format crate opts
into. A crate whose read path is a handful of small entries, or one called from
inside a caller's own parallel region, keeps the sequential readers and pays
nothing. [sfmtool-matches-format](../../crates/sfmtool-matches-format/)'s
[`read_matches`](../../crates/sfmtool-matches-format/src/read.rs) is the caller:
its sections between them consume every entry in the file, and its largest
columns are tens of megabytes.

The batch holds the whole decompressed file at once, where the sequential
readers hold one entry at a time. That is the cost of the shape and the reason
it is opt-in.

Errors are one enum rather than `Result<_, String>` so each format crate can map
variant to variant into its own public error (`SfmrError`, `MatchesError`,
`SiftError`, `CamRigError`, `KdfError`) and keep `ArchiveIoError` out of its public API.

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

Note the entry names: the crate treats them as opaque strings, so the
`{field}.{dims…}.{dtype}.zst` convention is enforced by each format crate
building its own names (`sfmtool-sfmr-format`'s and `sfmtool-matches-format`'s
`entries` modules do this from the counts in the metadata), and by the reader
passing the element count it expects.

## Implementation notes

**Alignment is the hazard the read path is built around.** A `Vec<u8>` returned
by the decompressor promises only 1-byte alignment, while reinterpreting it as
`u32`/`f32`/`f64` needs 4 or 8. `bytemuck::cast_slice` *panics* on a misaligned
buffer, so both read paths — `read_binary_array` (via its private `cast_or_copy`
helper) and the `raw_to_*` family — try the cast first and fall back to copying
through a freshly allocated, correctly aligned `Vec<T>`. The fast path is the
common one, because allocators over-align sizeable allocations; the fallback
exists so that correctness does not depend on the allocator's alignment. Which branch runs is not
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

**A batch read reports the first unreadable entry in archive order.** Decoding
runs concurrently, so "the error" would otherwise be whichever worker lost the
race, and a file with two corrupt entries would report differently from run to
run. `read_all` collects one `Result` per entry and reassembles them in the
order the ZIP lists, so the entry a caller is told about is a property of the
file.

**Decompressing in parallel is a scheduling change and nothing else.** A zstd
frame expands to the same bytes wherever it is expanded, so a format crate that
moves its read path onto `DecodedEntries` parses byte-identical arrays; the
shared conversion helpers behind `binary_array` and `uint128_array` are what
keeps the rest identical too.

**JSON bytes have one spelling.** `write_json_entry` stores exactly what
`json_entry_bytes` returns, which is `serde_json::to_vec`: compact, with no
trailing newline. A caller that needs a JSON entry's digest without writing an
archive (`sfmtool-sfmr-format`'s hash-only write path does) calls `json_entry_bytes`, so the
digest it computes is the one a written file would carry.

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

`DecodedEntries` is tested against the entry-at-a-time readers rather than on
its own terms, over an archive holding one of every kind of entry a format crate
reads — JSON, a `u32` column, an `f32` column compared by bit pattern, a digest
column, an empty one. Every buffer must equal what `read_zst_entry` returns for
the same entry, every typed accessor must equal its free-function counterpart,
and the length-mismatch and missing-entry errors must render the same string.
The archive-order rule gets a file with two unreadable entries, read repeatedly,
which must name the earlier one every time.

Two properties get dedicated tests because everything above rests on them:
`write_binary_entry_hashed` must produce the same digest as hashing the two
buffers by hand in write order *and* leave both entries readable in the archive,
and the alignment fallbacks must return the same values as the borrowing path for
a slice deliberately placed at an address congruent to 1 modulo the element
alignment (derived from the real base address, not by prepending a byte and
hoping). Whole-file round trips through real data live in each format crate's own
tests.

## Non-goals

- **No format schema.** The crate does not know entry names, required entries,
  versions or sections; each format crate supplies those. The one piece of
  metadata it defines is the `workspace` object (`WorkspaceMetadata` and
  `WorkspaceContents`), because `.sfmr` and `.matches` embed the same object and
  keep one definition of it here. `.kdf` defines its own
  `KdfWorkspaceMetadata`.
- **No verification.** Recomputing and comparing digests is each format crate's
  `verify_*` function, since only the format knows its sections and its
  structural constraints.
- **No streaming or partial writes.** Entries are compressed in one shot from a
  buffer the caller already holds, and a file is written whole.
- **No big-endian targets.** The crate fails to compile on a big-endian target
  rather than write byte-swapped columns; supporting one would mean swapping each
  element on read and write, which nothing implements.
