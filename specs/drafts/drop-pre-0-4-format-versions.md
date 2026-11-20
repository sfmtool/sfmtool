# Dropping the format versions older than sfmtool 0.4's (amendment)

**Status:** Draft

Amends [`../formats/format-versions.md`](../formats/format-versions.md), which
says that sfmtool 0.5 stops reading these versions and points here, and the
format specs it names: [`matches-file-format.md`](../formats/matches-file-format.md),
[`camrig-file-format.md`](../formats/camrig-file-format.md) and
[`kdf-file-format.md`](../formats/kdf-file-format.md), with
[`../cli/workspace/ws-upgrade-command.md`](../cli/workspace/ws-upgrade-command.md).

sfmtool 0.4.1's release notes say: "0.4.1 still reads the older `.sfmr`,
`.matches` and `.sift` versions so that `sfm ws upgrade` can convert them. The
next release will not." The next release is 0.5. This draft is that removal,
done as [format-versions.md § Dropping a version](../formats/format-versions.md#dropping-a-version)
says, one PR per format, with `.camrig` version 1 going in the same release.
`.sift` version 1, `.sfmr` versions 1 to 15 and the version 1 workspace file
are no longer read
([sift-file-format.md § Format versions](../formats/sift-file-format.md#format-versions),
[sfmr-file-format.md § Format versions](../formats/sfmr-file-format.md#format-versions),
[workspace.md § Versions](../workspace/workspace.md#versions)), and
`sfm ws upgrade` refuses a version 1 workspace; this draft holds what is still
to do: `.matches` versions 1 to 9 and `.camrig` version 1.

## What stops being read

sfmtool 0.4 and 0.4.1 write `.sfmr` 16, `.matches` 10, `.sift` 2, `.camrig` 2
and `.kdf` 3, so 0.5's read window is exactly those versions.

| Format | Current | Oldest read today | Oldest read in 0.5 | Last release that reads the dropped versions |
|--------|---------|-------------------|--------------------|----------------------------------------------|
| `.matches` | 10 | 1 (6 with clusters) | 10 | 0.4.1 |
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
file below the oldest version read with one message, as the `.sift` and
`.sfmr` readers already refuse an older file (`SiftError::VersionNoLongerRead`,
`SfmrError::VersionNoLongerRead`):

```
<path> is a version 7 .matches file, and this sfmtool reads .matches files of
version 10 or later. sfmtool 0.4.1 is the last release that reads version 7:
install it, run `sfm ws upgrade` on the workspace, then use the file with this
release.
```

The `.camrig` message is the same with its own numbers: a version 1 `.camrig`
file, read by version 2 or later, and sfmtool 0.4.1 named as the last release
that reads it.

The format crate holds the oldest version read beside its current-version
constant (`MATCHES_OLDEST_READ_VERSION`, `CAMRIG_OLDEST_READ_VERSION`), and
the release named in the message as a constant beside it
(`MATCHES_LAST_RELEASE_READING_OLDER_VERSIONS`,
`CAMRIG_LAST_RELEASE_READING_OLDER_VERSIONS`), as the `.sift` and `.sfmr`
crates hold `SIFT_OLDEST_READ_VERSION` and `SFMR_OLDEST_READ_VERSION` with
their `*_LAST_RELEASE_READING_OLDER_VERSIONS`. Each crate's tests check the refusal of
the version just below the oldest read, and of a version above the current
one; the tests that read each dropped version go.

## `sfm ws upgrade`

`sfm ws upgrade` rewrites each `.matches` and `.camrig` file below the current
version that the readers accept, and lists a `.sfmr` below version 16 as one
the reader refuses
([ws-upgrade-command.md](../cli/workspace/ws-upgrade-command.md)). Each PR
below removes its format's older versions from what the upgrade rewrites, so
once both land it rewrites none until a format next changes version, and
`ws-upgrade-command.md` is updated in each PR to say which it still rewrites.

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
- `sfm ws upgrade`'s handling of a cluster file below version 6 (the cluster
  check of `plan_matches` in `workspace/upgrade/plan.rs`), which the general
  refusal replaces, and its filling of a version 3 or older file's image
  dimensions from the `.sift` files (`rewrite_matches` in
  `workspace/upgrade.rs`);
- the tests that read each older version.

No `.matches` file is checked in. The spec is trimmed as the `.sfmr` one was
([sfmr-file-format.md § Format versions](../formats/sfmr-file-format.md#format-versions)):
"Versioning and Migration" becomes a "Format versions" section near the top,
giving version 10 as both the current and the oldest version read, every
"(version N+)" tag goes, and "Version History" stays at the end, each line
standing alone.

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
