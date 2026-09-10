# Validation tests for the KDF format

**Status:** Draft. Amends
[core/features/lazy-kdforest-query.md](../core/features/lazy-kdforest-query.md)
§ "How parity is tested" and
[formats/kdf-file-format.md](../formats/kdf-file-format.md).

The `.kdf` reader, writer and verifier reject roughly a hundred distinct
malformed inputs, and the current suite reaches almost none of them. Parity and
laziness are well covered — that is where the design risk was — but a reader
that accepts a cyclic tree, or a verifier that misses a duplicated feature ID,
would today pass every test in the workspace. This draft proposes the negative
suite that closes that gap.

## Why it needs its own harness

Every structural check in
[read.rs](../../crates/sfmtool-kdf-format/src/read.rs) runs *after* the chunk's
XXH128 digest is verified. Mutating a node column in a written file therefore
produces `Integrity`, not the structural error under test, and a test that
asserts merely "this is an error" would pass whether or not the check it names
still exists.

So the harness has to rebuild the hash directory after it mutates. It needs one
helper that reopens a `.kdf`, hands each entry's *decoded* bytes to a callback
that may replace or drop them, recomputes the chunk, section and whole-file
digests in the order
[kdf-file-format.md](../formats/kdf-file-format.md) § "Hash composition" fixes,
and rewrites the archive. With that in place each case is a few lines, and every
case asserts a specific `KdfError` variant and message rather than `is_err()`.

## What to cover

**Structure.** A child reference whose chunk, local index or repeated logical ID
is out of range; a repeated logical ID that disagrees with the addressed node; a
cycle and a shared child, both of which the reachability pass in
[verify.rs](../../crates/sfmtool-kdf-format/src/verify.rs) owns; leaf starts that
are not contiguous in local node order; a leaf range running past the chunk's
feature rows; a nonzero value in a field the spec reserves as zero; an unknown
node kind.

**Entry set and metadata.** A missing entry, a duplicate, and one that belongs to
the other descriptor layout; an unsupported `version`; a scalar type that does
not match the caller's; a declared `decoded_bytes` that disagrees with the
arrays; a truncated zstd frame.

**Whole-file invariants.** The verifier's own checks: a duplicated or missing
feature ID in a tree's permutation, vectors for one feature ID that differ
between two trees, and a descriptor that falls on the wrong side of an ancestor's
split plane.

**Numerics.** A nonfinite `f32` split coordinate and a nonfinite stored vector,
both of which must be refused rather than propagated into a distance.

**Sources.** `verify_sift_sources` has no test at all. It needs fixtures written
through `sift-format`: a matching workspace, a relocated one, a missing file, a
changed content hash, an origin naming an image index that does not exist, and a
duplicated `(image_index, image_feature_index)` pair. A query must keep
succeeding when the `.sift` files are gone, since the origins table is embedded.

**Reads.** That traversing an unvisited chunk reads none of its bytes, and that a
warm resident hit costs no read and no decode. Both are already observable
through `io_stats`; nothing asserts them yet.

## Out of scope

ZIP64 offsets, which need multi-gigabyte fixtures, and a file-to-file layout
converter, which does not exist — re-exporting from the in-memory forest is
today the only way to change layout.
