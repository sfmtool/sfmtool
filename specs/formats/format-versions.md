# File format versions

Each of sfmtool's five on-disk formats (`.sfmr`, `.matches`, `.sift`,
`.camrig` and `.kdf`) records a format version in its `metadata.json`, and
this spec says how that version changes. It says when a change to a format
needs a new version and when it does not, what a version change consists of
(the constant, the writer, the reader, `sfm ws upgrade`, the tests, the
checked-in test data, the format spec and the release notes), which older
versions a reader accepts, how a version stops being read, and how a format
spec presents its versions. It holds for all five formats; each format spec
says what its own versions are.

## Where the versions are

Each format crate has one constant for the version its writer writes:

| Format | Constant | Current version | Oldest version read |
|--------|----------|-----------------|---------------------|
| `.sfmr` | [`SFMR_FORMAT_VERSION`](../../crates/sfmtool-sfmr-format/src/types.rs) | 17 | 16 |
| `.matches` | [`MATCHES_FORMAT_VERSION`](../../crates/sfmtool-matches-format/src/types.rs) | 10 | 10 |
| `.sift` | [`SIFT_FORMAT_VERSION`](../../crates/sfmtool-sift-format/src/types.rs) | 2 | 2 |
| `.camrig` | [`CAMRIG_FORMAT_VERSION`](../../crates/sfmtool-camrig-format/src/types.rs) | 2 | 2 |
| `.kdf` | [`KDF_FORMAT_VERSION`](../../crates/sfmtool-kdf-format/src/types.rs) | 3 | 3 |

A version at which something changed that a reader gates on is a constant of
its own in the format crate, so a later version change does not move the gate,
as `REFINED_SHAPE_GATES_VERSION` was for `.matches`
([worked example](#worked-example-matches-version-9--version-10)). A format
whose readers have stopped reading an older version holds, beside its version
constant, the oldest version read and the last release that reads the older
ones, which its refusal names ([Dropping a version](#dropping-a-version)):
`SFMR_OLDEST_READ_VERSION` and `SFMR_LAST_RELEASE_READING_OLDER_VERSIONS` for
`.sfmr`, `MATCHES_OLDEST_READ_VERSION` and
`MATCHES_LAST_RELEASE_READING_OLDER_VERSIONS` for `.matches`,
`SIFT_OLDEST_READ_VERSION` and `SIFT_LAST_RELEASE_READING_OLDER_VERSIONS` for
`.sift`, and `CAMRIG_OLDEST_READ_VERSION` and
`CAMRIG_LAST_RELEASE_READING_OLDER_VERSIONS` for `.camrig`.

The `.sfmr` readers stop at version 16, the `.matches` readers at version 10
and the `.sift` and `.camrig` readers at version 2, the versions sfmtool 0.4
writes. sfmtool 0.4.1's release notes announced that the next release stops
reading the older `.sfmr`, `.matches` and `.sift` versions; `.camrig` version
1 is one no release wrote, so it was dropped with no announcement. Every
release that writes `.kdf` files has written version 3, and no release has
read `.kdf` versions 1 and 2, so the `.kdf` crate holds no oldest version read
and no last release reading older versions
([kdf-file-format.md § Format versions](kdf-file-format.md#format-versions)).

## When a format's version changes

A format's version changes when a reader of the previous version, given a file
written under the change, would refuse it, misread it, or silently drop or
misinterpret something it holds. The new version makes that reader refuse the
file as newer than it reads, which is the safe outcome, and tells the new
reader which rules an older file follows. The cases met so far:

- **A new required entry or key.** `.sfmr` version 15 requires each image's
  sharpness and gradient structure tensor, and `.sift` version 2 requires
  `image_to_gray` and the same two readings. The version tells a reader
  whether a file must have them.
- **A new optional entry.** In this project a new optional entry also changes
  the version. A reader that does not know an entry reads the file as if it
  were absent: one of `.sfmr` version 12 would read a version 13 file's
  images through their cameras alone, ignoring the per-image focal and
  principal-point adjustments, and so place them wrongly, and it would drop
  `tracks/observation_confidence` (version 6) on a re-save. Each optional
  `.sfmr` column since version 6 came with a version change; the one added
  while a version was current, `points3d/normal_confidence` in version 5,
  left a version 5 file that may or may not carry it. A writer writes the
  current version whether or not a file carries the optional entries.
- **A new value of an enumeration or legend.** `.matches` version 10 lets the
  member-status legend name two new statuses. A version 9 reader refuses a
  legend name it does not define, and without the change it would meet them
  in a file claiming a version it reads ([worked example](#worked-example-matches-version-9--version-10)).
- **A changed meaning, unit, data type or entry name.** `.sfmr` version 16
  stores the sharpness and five observation-reading columns as `float16` and
  the two bitmap scores as their ZNCC deficits `1 − z`; `.sfmr` version 5,
  `.matches` version 2 and `.camrig` version 2 changed the stored poses to the
  canonical camera convention with no change of layout. A changed meaning
  needs the change even when every entry keeps its name and layout: the
  `.sfmr` `points3d/reprojection_errors` column changed from the mean of the
  observation errors to their RMS while version 12 was current, with no
  version change, so a reader cannot tell which a version 12 file holds, and
  the format spec has to say so column by column.
- **A change to what a hash covers.** `.sfmr` version 8 moved the write time
  out of `metadata.json` and so out of `content_xxh128`, and version 10 moved
  the depth statistics into a `derived/` section outside `content_xxh128`.
  A reader that verified a file under the other version's rules would refuse
  a sound file, or pass a broken one.

The version does not change for a change no reader of the previous version
can observe: a writer that computes a value a better way under the same
definition, or retiring an optional key that current readers skip at every
version, as `.sfmr` retired the version 9 `lineage` key with no version change
([sfmr-file-format.md § The retired `lineage` key](sfmr-file-format.md#the-retired-lineage-key)).

## What a version change consists of

One change, landed together:

1. **The constant.** The format crate's version constant goes up by one.
2. **The writer** writes only the current version. It does not choose the
   version by which optional entries a file carries.
3. **The reader** accepts the new version and every older version still in the
   [read window](#the-read-window), and converts an older file into the
   current in-memory form inside the format crate, at the I/O boundary. No
   caller branches on a file's version. Where the current form holds data an
   older file lacks and the file alone cannot supply it, the in-memory form
   says the data is absent, and the writer refuses to write the current
   version until a caller supplies it, as the sfmtool 0.4 reader did for a
   `.sfmr` file below version 15, which lacks image readings.
4. **`sfm ws upgrade`** brings every version the reader accepts to the current
   one ([../cli/workspace/ws-upgrade-command.md](../cli/workspace/ws-upgrade-command.md)).
   For `.sfmr`, `.matches` and `.camrig`, which it rewrites by loading and
   saving, this follows from the reader and the writer. A `.sift` change needs
   a lossless conversion of its own, as sfmtool 0.4.1's upgrade converted
   version 1 to version 2, because extracting the features again does not
   reproduce them and every file that records their indexes and hashes would
   break. Every `.kdf` file a release has written is at the current version,
   so the upgrade reads none today; the change that next moves the `.kdf`
   version makes the upgrade rewrite a version 3 index as the current version,
   as it rewrites the other formats.
5. **Tests** read a file of each accepted older version and check what it
   converts to, and check that the reader refuses a version newer than the
   current one and, where the oldest version read is above 1, a version below
   it.
6. **The checked-in files** under `test-data/` are rewritten at the current
   version, keeping their names: the `_vNN` in a file's name versions the
   reconstruction it holds, not its format (`AGENTS.md`, under `test-data/`).
7. **The format spec** is updated as
   [How a format spec presents its versions](#how-a-format-spec-presents-its-versions)
   says.
8. **The release notes** of the release that ships it say which format
   changed, what changed, and that `sfm ws upgrade` converts older files.

## The read window

A reader accepts the current version and every version the previous release
wrote. That is the **read window**: a file written by the release before
this one always reads, and `sfm ws upgrade` in this release converts it. A
reader may read versions older than the window until a release drops them. A
version no release wrote, one only development builds wrote between two
releases, lies outside every release's window.

## Dropping a version

A version below the read window that some release wrote is dropped in two
steps: one release's notes announce that the next release stops reading it,
and the next release removes it, so a user of the announcing release learns
in time to upgrade the files that release wrote. A version no release wrote
has no such users, so any release may stop reading it with no announcement.
Removing a version deletes its read path and conversion, its tests, and its
subsection of the format spec's Format versions section, and raises the
oldest version read. A reader refuses a file of a dropped version that some
release read with a message that names the file, its version, the oldest
version it reads, and the last sfmtool release that reads the file's version,
so the user can install that release and run `sfm ws upgrade` with it. A
version no release read, such as version 0 or `.kdf` versions 1 and 2, has no
release to name, and is refused as a version newer than the current one is.

## How a format spec presents its versions

A format spec has a **Format versions** section near the top, after its design
principles, so a reader knows which version the entries describe before
reading them. It gives the current version, the oldest version read, and the
rule a reader applies to a version outside that range, and has a
"Version N → Version N+1" subsection for each older version still read,
saying what differs and how the reader converts it. It links this spec.

An entry or key that some version still read lacks is tagged with the version
that introduced it, such as "(Optional, version 13+)". The tag goes when no
version still read lacks it.

A **Version History** section ends the spec: one line per version, newest
first, saying what it changed. It is permanent. When a version stops being
read its migration subsection goes and its history line stays, so the
history records every version a file in the wild may hold.

## Worked example: `.matches` version 9 → version 10

The `.matches` version 10 change, which sfmtool 0.4 shipped, added two member
statuses, `rejected_unlocalizable_refined` and `rejected_unlocalizable_cells`,
codes 7 and 8, to the names the `cluster_patches/metadata.json`
`member_status_names` legend may give, and two recorded settings to
`refine_options`. No entry changed its layout.

- **Why it changed the version.** A writer states the whole legend, so every
  new file with a `cluster_patches/` section names the new statuses, and a
  version 9 reader refuses a legend name it does not define. Without the
  change, such a file would claim a version that reader reads and then fail
  on a name; with it, the reader refuses the file as newer than it reads.
- **The constant.** `MATCHES_FORMAT_VERSION` became 10, and a constant of its
  own, `REFINED_SHAPE_GATES_VERSION` (10), recorded where the new names start.
- **The reader** read a version 9 file unchanged, and refused a version 9
  file whose legend named either new status, since no version 9 writer wrote
  them. A test relabelled a written file as version 9 and checked that
  refusal.
- **The spec** gained a "Version 9 → Version 10" subsection, with a table of
  what changed and a paragraph on how a version 9 file reads and why the
  version changed, and a Version History line for version 10.
- **Dropping version 9.** sfmtool 0.4.1's release notes announced that the
  next release stops reading the older `.matches` versions, and sfmtool 0.5
  did: its reader refuses a version 9 file with the message naming 0.4.1,
  `REFINED_SHAPE_GATES_VERSION` and the subsection went, and the Version
  History line stayed
  ([matches-file-format.md § Version History](matches-file-format.md#version-history)).
