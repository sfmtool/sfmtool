# Dropping the format versions older than sfmtool 0.4's (amendment)

**Status:** Draft

Amends [`../formats/format-versions.md`](../formats/format-versions.md), which
says that sfmtool 0.5 stops reading these versions and points here, and the
format specs it names: [`sfmr-file-format.md`](../formats/sfmr-file-format.md),
[`matches-file-format.md`](../formats/matches-file-format.md),
[`sift-file-format.md`](../formats/sift-file-format.md),
[`camrig-file-format.md`](../formats/camrig-file-format.md) and
[`kdf-file-format.md`](../formats/kdf-file-format.md), with
[`../cli/workspace/ws-upgrade-command.md`](../cli/workspace/ws-upgrade-command.md)
and [`../workspace/workspace.md`](../workspace/workspace.md).

sfmtool 0.4.1's release notes say: "0.4.1 still reads the older `.sfmr`,
`.matches` and `.sift` versions so that `sfm ws upgrade` can convert them. The
next release will not." The next release is 0.5. This draft is that removal,
done as [format-versions.md § Dropping a version](../formats/format-versions.md#dropping-a-version)
says, one PR per format, with `.camrig` version 1 and the version 1 workspace
file going in the same release.

## What stops being read

sfmtool 0.4 and 0.4.1 write `.sfmr` 16, `.matches` 10, `.sift` 2, `.camrig` 2
and `.kdf` 3, so 0.5's read window is exactly those versions.

| Format | Current | Oldest read today | Oldest read in 0.5 | Last release that reads the dropped versions |
|--------|---------|-------------------|--------------------|----------------------------------------------|
| `.sfmr` | 16 | 1 | 16 | 0.4.1 |
| `.matches` | 10 | 1 (6 with clusters) | 10 | 0.4.1 |
| `.sift` | 2 | 1 | 2 | 0.4.1 |
| `.camrig` | 2 | 1 | 2 | 0.4.1 |
| `.kdf` | 3 | 3 | 3 | — |

**`.camrig` version 1** was written by no release: the format came in after
0.2 and moved to version 2 before 0.3, so only development builds wrote
version 1. It lies outside every release's read window, so 0.5 stops reading
it with no announcement
([format-versions.md § Dropping a version](../formats/format-versions.md#dropping-a-version)).
0.4.1's `sfm ws upgrade` rewrites it as version 2.

**`.kdf`** follows the same read window. Every release, 0.3 to 0.4.1, wrote
version 3, so 0.5's window is version 3 alone, which is what its reader
already reads (versions 1 and 2 were written only by development builds).
Nothing is dropped. Its spec's "Version" subsection still becomes a "Format
versions" section near the top, with a Version History at the end, in a PR
of its own, so all five specs share the shape.

## The refusal

Each format crate's reader, `read_<fmt>_metadata` and `verify_<fmt>` refuse a
file below the oldest version read with one message:

```
<path> is a version 12 .sfmr file, and this sfmtool reads .sfmr files of
version 16 or later. sfmtool 0.4.1 is the last release that reads version 12:
install it, run `sfm ws upgrade` on the workspace, then use the file with this
release.
```

The `.camrig` message is the same with its own numbers: a version 1 `.camrig`
file, read by version 2 or later, and sfmtool 0.4.1 named as the last release
that reads it.

The format crate holds the oldest version read beside its current-version
constant (for example `SFMR_OLDEST_READ_VERSION`), and the release named in
the message as a constant beside it. Each crate's tests check the refusal of
the version just below the oldest read, and of a version above the current
one; the tests that read each dropped version go.

## `sfm ws upgrade` and the workspace file

0.5's `sfm ws upgrade` upgrades from a version 2 workspace. It refuses a
version 1 `.sfm-workspace.json` before reading anything else, with a message
that names the workspace file, says it is version 1, and says to upgrade it
with sfmtool 0.4.1 first, since 0.4.1 is the last release that reads the
older `.sift`, `.sfmr`, `.matches` and `.camrig` versions it would have to
convert. Every other command's refusal of a version 1 workspace
([workspace.md § A command that uses a workspace requires version 2](../workspace/workspace.md#a-command-that-uses-a-workspace-requires-version-2))
names sfmtool 0.4.1 the same way, in place of running `sfm ws upgrade`.

What goes from the upgrade: its rewrite of a version 1 workspace file (phase
5), its conversion of version 1 `.sift` folders and the deletion of the old
folders (phases 3 and 6), and its deletion of a `.kdf` index built from a
version 1 record. What stays: rewriting any file below the current version
that the 0.5 readers accept, which is none until a format next changes
version. `ws-upgrade-command.md` is rewritten to describe that, and
`workspace.md` loses the version 1 workspace's rules ("Reading version 1 files
stays allowed" and the version 1 column of its versions table) except the
refusal, keeping a history line for the workspace file's version 1.

## `.sfmr`: versions 1 to 15 (one PR)

Goes:

- every branch of `read_sfmr`, `read_sfmr_metadata` and `verify_sfmr` on a
  version below 16: `positions_xyz` and `points3d_indexes` (version 1),
  `estimated_normals_xyz` (2), the implicit `sift_files` source (below 4), the
  timestamp in `metadata.json` (below 8), the depth statistics in `images/`
  (below 10), required thumbnails (below 11), the absent reference observations
  (below 12), the absent per-image adjustments (below 13 and 14), the absent
  image readings (below 15) and the `float32` reading and score columns and
  stored normals with the patch frame (15), with the version constants that
  gate only those (`SFMR_CANONICAL_CONVENTION_VERSION`,
  `SFMR_WRITE_RECORD_VERSION`, `SFMR_REFERENCE_OBSERVATIONS_VERSION`,
  `SFMR_PER_IMAGE_ADJUSTMENTS_VERSION`,
  `SFMR_PER_IMAGE_LENS_CORRECTIONS_VERSION`, `SFMR_IMAGE_READINGS_VERSION`,
  `SFMR_DERIVED_PATCH_NORMALS_VERSION`, `SFMR_FLOAT16_READINGS_VERSION`);
- the COLMAP-to-canonical conversion of a file below version 5 in
  `sfmtool-core`'s
  [`reconstruction/data/conversion.rs`](../../crates/sfmtool-core/src/reconstruction/data/conversion.rs),
  which today runs outside the format crate;
- the tests that read each older version.

The save's gathering of image readings stays: it also fills the readings of a
reconstruction a solve or an import builds in memory.

Every checked-in `.sfmr` file under `test-data/` is at version 16 already,
and every checked-in workspace file at version 2, as
[format-versions.md § What a version change consists of](../formats/format-versions.md#what-a-version-change-consists-of)
item 6 requires, so the PR rewrites none of them.

### The spec

"Versioning and Migration" becomes a "Format versions" section near the top,
after "Design Principles", giving version 16 as both the current and the
oldest version read, with no migration subsections. Every "(version N+)" tag
goes ("(Optional, version 13+)" becomes "(Optional)"), as do "Reading older
files", the migration note and the below-version-5 sentence in "Conversions
happen at the I/O boundary", and each column's account of what writers of
older versions stored, such as `points3d/reprojection_errors`'s mean-or-RMS
list. "Version History" stays, each line made to stand alone in place of its
link to a removed subsection.

## `.matches`: versions 1 to 9 (one PR)

Goes:

- the branches of `read_matches`, `read_matches_metadata` and
  `verify_matches` on a version below 10: the `S`-conjugation of version 1
  relative poses, the absent `has_clusters` flags (below 3), the absent
  `image_dims` (3 and below), the cluster-backbone refusal below 6, the
  absent `member_status_names` legend (6), the absent per-cell entries
  (below 8) and the legend names a version may not hold (below 9 and 10),
  with `MEMBER_STATUS_LEGEND_VERSION`, `MEMBER_CELLS_VERSION`,
  `REFUSED_OUTLIER_VERSION` and `REFINED_SHAPE_GATES_VERSION`;
- `image_dims` as an `Option` in memory and in the binding
  ([`fileio/matches.rs`](../../crates/sfmtool-py/src/fileio/matches.rs),
  [`fileio/matches_file.rs`](../../crates/sfmtool-py/src/fileio/matches_file.rs)),
  and `sfm estimate-intrinsics`'s message for a file with no dimensions;
- `sfm ws upgrade`'s handling of a cluster file below version 6
  (`unreadable_clusters` in `workspace/upgrade/plan.rs`), which the general
  refusal replaces, and its filling of a version 3 or older file's image
  dimensions from the new `.sift` files;
- the tests that read each older version.

No `.matches` file is checked in. The spec is trimmed as for `.sfmr`, keeping
"Version History".

## `.sift`: version 1 (one PR)

Goes:

- the version 1 branches of the reader (`SIFT_READINGS_VERSION`): a version 1
  file's absent `image_to_gray` and readings;
- every caller's branch on a version 1 file, which today sits outside the
  format crate: `sfmtool-core`'s `analysis/pick_images/frames.rs`,
  `features/sift/files.rs` and `reconstruction/image_readings.rs`;
- `Version2SiftFolder` and `upgrade_feature_options`, the conversion of a
  version 1 file, and the `version_1_sift_files` check every writer makes
  before writing into a folder ([workspace.md § Versions never mix](../workspace/workspace.md#versions-never-mix));
- `sfm ws upgrade`'s conversion of version 1 folders and of a version 1
  workspace file, as [above](#sfm-ws-upgrade-and-the-workspace-file), and the
  viewer's in-memory conversion of a version 1 record in *Build SIFT Files*;
- the tests that read or convert version 1 files.

No `.sift` file is checked in. The spec's "Format versions" section loses
"Converting a version 1 file" and the table of the two versions; "Version
History" stays.

## `.camrig`: version 1 (one PR)

Goes:

- the reader's acceptance of version 1 and its upgrade of the version 1
  poses to the canonical convention (`S`-conjugation for a body-anchored rig,
  a left `S` multiply for a `spherical_tiles` rig), with
  `CAMRIG_CANONICAL_CONVENTION_VERSION`;
- `sfm ws upgrade`'s rewrite of a `.camrig` file below version 2;
- the tests that read version 1 files.

The one checked-in `.camrig` file is at version 2. The spec's "Versioning and
migration" becomes a "Format versions" section near the top, after "Format
design principles", giving version 2 as both the current and the oldest
version read, without the "Version 1 → Version 2" subsection; "Version
history" stays.
