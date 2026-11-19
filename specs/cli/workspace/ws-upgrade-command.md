# `sfm ws upgrade` Command

`sfm ws upgrade` brings a workspace made by sfmtool 0.3 or a development
build, and every file in it, to the current file versions. It converts the
workspace's version 1 `.sift` files to version 2 in new features folders,
without extracting them again; rewrites each `.sfmr` and `.matches` file at its
own path to name the new folder and the new files' content hashes, and each
version 1 `.camrig` file as version 2; deletes each `.kdf` index built from an
old folder; rewrites the version 1 `.sfm-workspace.json` as version 2, keeping
every option; and deletes the old features folders. After it, every file
names version 2 `.sift` files, so nothing made before the upgrade is lost when
readers drop the format versions older than those sfmtool 0.4 writes (`.sfmr`
16, `.matches` 10, `.sift` 2, `.camrig` 2): sfmtool 0.5 drops them all, and
its `sfm ws upgrade` refuses a version 1 workspace, as
[../../drafts/drop-pre-0-4-format-versions.md](../../drafts/drop-pre-0-4-format-versions.md)
specifies. It rewrites files in place and
deletes folders, so it cannot be undone: **back up the workspace first**
([Backing up](#backing-up)). Every other command that uses a workspace refuses
a version 1 one and says to run this command first
([workspace.md § A command that uses a workspace requires version 2](../../workspace/workspace.md#a-command-that-uses-a-workspace-requires-version-2)).
The two versions of the workspace file are in
[workspace.md § Versions](../../workspace/workspace.md#versions).

## Command Syntax

```bash
sfm ws upgrade [PATH] [--dry-run]
```

The workspace is found by the usual upward search for `.sfm-workspace.json`
from `PATH`, or from the current directory when `PATH` is omitted. The run
upgrades the whole **tree** of workspaces that workspace belongs to: the
outermost workspace at or above it, the tree's root, and every workspace
nested in that one. One run does the whole tree because a file of one
workspace can name the `.sift` files of another, in either direction, and two
runs that each decided what to delete without seeing the other's files could
each delete what the other still needs. When `PATH` is in a workspace nested
in another, the command refuses before reading anything else, names the
outermost workspace and says to run the upgrade there.

`--dry-run` finds the files and plans their conversion (phases 1 and 2 below),
prints the same report a run would, and writes nothing.

When no file needs converting and no version 1 features folder is left to
delete, the command says "nothing to do" and changes nothing. On a tree whose
workspace files are all version 2 it still runs every phase that has work, so
a run that stopped part way can be run again to finish, and a version 2
workspace that holds older files is brought up to date.

## Backing up

The command deletes the old features folders and rewrites `.sfmr`, `.matches`
and `.camrig` files in place, so it cannot be undone. Its help, this spec, and
the first line it prints before converting anything say to back up the
workspace first. Saving changes each `.sfmr` file's `content_xxh128`, so a
point ID (`pt3d_<hash>_<index>`) written down from a `.sfmr` before the upgrade
no longer resolves; the report says so when it rewrote a `.sfmr`.

## Why the `.sift` files are converted, not extracted again

Extracting again does not reproduce the version 1 features. For the `sfmtool`
tool the grey conversion moved from `f64` to `f32` after version 2 was
introduced, and 0.04% of keypoints appear on one side only and 2.8% of
descriptors differ; for `colmap` the result depends on the pycolmap build and
the GPU. A `.sfmr` or `.matches` file records feature indexes into its `.sift`
files, so new features would invalidate every one of them.

A version 1 file converts to version 2 losslessly
([sift.md § Converting a version 1 `.sift` file](../../core/features/sift.md#converting-a-version-1-sift-file)):
positions, affine shapes, descriptors and the thumbnail are copied byte for
byte; `feature_options` becomes `upgrade_feature_options(feature_tool,
feature_options)`; the sharpness and gradient structure tensor are computed
from the image file through the converted options' `image_to_gray`, after the
image is verified against the file's `image_file_xxh128`; and
`metadata.version` becomes 2. Both of the file's hashes change
([sift-file-format.md](../../formats/sift-file-format.md)), which is why every
file that records them is rewritten too.

No version 1 record carries `double_image_below_long_side`, and
`upgrade_feature_options` adds none, so an upgraded workspace keeps doubling
every image as its features were extracted (the standard version 1 `sfmtool`
options convert to `features/sift-sfmtool-9017ec0d2c5dfc401da610578062b53d`).
Adopting the 4096-pixel limit `sfm ws init` records takes `sfm ws init --force`
and extraction into a new folder.

## What it does

The command runs these phases in order.

1. **Find the files.** It walks the tree for `.sfmr`, `.matches`, `.camrig`
   and `.kdf` files and for the `.sift` folders named by a
   `feature_prefix_dir`: each workspace file's own, and each one recorded in a
   `.sfmr` or `.matches` file's `workspace.contents`. A `feature_prefix_dir` is
   relative to each image's directory, so one record names one folder in every
   directory that holds images
   ([workspace.md § Feature Storage Convention](../../workspace/workspace.md#feature-storage-convention)),
   and a workspace can hold several records' folders, left by earlier
   `sfm ws init --force` runs. The walk also finds every version 1 features
   folder whether or not a record names it: each directory that holds `.sift`
   files is checked by the rule `version_1_sift_files` applies, the same check
   every writer of version 2 files makes before it writes into a folder, read
   with `scan_sift_folder`. A `.sift` file the file system will not give up
   (locked, or a sync client's placeholder that cannot be fetched) stops the
   run before anything is written, listing each such file and its error, since
   it could hold features a file needs; one whose content the reader rejects is
   passed over, as the writers pass it over. A version 1 folder that no file
   and no workspace file of the tree names is an **orphan**: it is not
   converted, and phase 6 deletes it with the others.

   The walk enters nested workspaces, but not a symbolic link or junction to a
   directory, whose target may lie outside the tree; the report lists each.
   Every file is attributed to the workspace its `workspace` record resolves to
   (by the rule every reader resolves it with), and its image names and
   folders are relative to that workspace. The folders a file names are
   `<its workspace>/<image's directory>/<feature_prefix_dir>` for each image it
   records, read from the file's record and image names alone with
   `read_sfmr_image_names` and `read_matches_image_names`, so a file the
   current reader refuses still names its folders. A file another tool wrote,
   or one edited by hand, can hold a name that breaks the image-name rule
   ([sfmr-file-format.md § `images/names.json.zst`](../../formats/sfmr-file-format.md#imagesnamesjsonzst)),
   such as one with `..`, and the reader refuses such a file. When a cheap
   reader refuses its names, as both do when their count differs from the
   record's `image_count`, they are read from the file's names entry without
   that reader's checks (`read_sfmr_image_names_unchecked`,
   `read_matches_image_names_unchecked`). Each `..` of a name is resolved by
   name, so that the folders the file names are still found and kept. A file
   whose image names cannot be read, are refused by a cheap reader, or break
   the image-name rule also names the folders its record's
   `feature_prefix_dir` names below its workspace.
   Every folder, image directory and `.sift` file is identified by its real
   path (`std::fs::canonicalize`) and spelled as the walk spells it, so a file
   that names `IMAGES/<image>`, as `sfm sift --extract IMAGES` records on a
   file system that ignores case, names the same folder as the walk's
   `images`.

   Each `.sfmr` and `.matches` file the run would rewrite is read whole here,
   in a dry run too, so a file the current reader refuses is told apart, by the
   error's type, from one the file system will not give up (locked, or it
   cannot be opened or read), which **fails**. A file whose metadata cannot be
   read fails too; one whose names entry does not read as a list of names
   names its record's prefix folders instead. A `.sift` file a file records that is
   not on disk, or is not the version 1 file it records, is found here, before
   anything is written.
2. **Map each old folder to a new one.** A version 1 folder is named by a file
   when it is one of the folders phase 1 found for it, and by a version 1
   workspace file's record (`feature_tool`, `feature_type`, `feature_options`,
   `feature_prefix_dir`) when that workspace is the deepest of the tree the
   folder lies in, the folder is `<image dir>/<feature_prefix_dir>` below it,
   and, when files name the folder, `<image dir>` is the directory of their
   images. So a workspace file names no folder of a workspace nested in its
   own, and a prefix that ends another (`sift-x` beside `features/sift-x`) is
   not taken for it where files show which one the folder is. Several records
   can name one folder: a workspace file of sfmtool 0.3 records
   `feature_type: "sift-sfmtool"` and its `.sfmr` files `"sift"`, or two files
   record options that differ. Each physical old folder therefore maps to
   exactly one new folder, `<image dir>/<new feature_prefix_dir>`, and one
   converted record, named by `feature_prefix_dir`:
   - when a version 1 workspace file's own record names the folder, the record
     that workspace file is converted to in phase 5;
   - otherwise the folder's own `.sift` files decide: the record of their
     feature tool, the tool's workspace `feature_type` (`feature_type_for_tool`,
     the `sift-<tool>` form `sfm ws init` records), and their
     `feature_options` converted. When they record different tools or options,
     the run stops here, naming the folder and two files that differ.

   Every file that names the folder, whatever its record, is pointed at that
   one new folder and takes that one converted record, so the records come out
   the same. A file that names folders that map to different new folders stops
   the run, naming the file and the folders. A file that names no folder on
   disk takes its workspace file's converted record when that names the same
   `feature_prefix_dir`, and otherwise its own record converted under the
   tool's workspace `feature_type`. A record that cannot be converted (an
   `sfmtool-undistort` record, whose files only `sfm undistort` makes; a tool
   this build does not know; a `gray_formula` that does not parse) stops the
   run, naming the files that hold it, since its folder could be neither
   converted nor deleted. So does a record whose `feature_prefix_dir` is empty,
   absolute or holds `..`, and a version 1 workspace file whose options already
   carry `image_to_gray`. Two old folders whose records convert to the same new
   folder, such as a hand-named `features/sift-colmap-3000` beside a hashed one
   with the same options, are converted into it together: a file the two hold
   with the same content hash is converted once, and two different files for
   one image stop the run, naming both.
3. **Convert the `.sift` files** of every old folder that a version 1
   workspace file or a file this run rewrites names, into its new folder, in
   parallel. A folder named only by files the run leaves is not converted,
   since none of them would read the new files. The image of
   `<image dir>/<feature_prefix_dir>/<name>.sift` is `<image dir>/<name>`,
   never the file's `image_name`, which version 1 extractors sometimes wrote
   relative to another directory. An image whose file is missing, cannot be
   decoded or no longer matches `image_file_xxh128` cannot be converted; every
   such `.sift` file is listed and the run stops at the end of this phase,
   before any other file is rewritten. A new file already in the new folder is
   kept when it is this file's conversion, as a rerun finds it, and is
   otherwise refused the same way rather than replaced, since it may be an
   extraction other files record.
4. **Rewrite the files**, each atomically at its own path (a temporary file
   beside the target, renamed over it):
   - a `sift_files` `.sfmr` that names an old folder: `workspace.contents`
     becomes the converted record and `feature_tool_hashes` and
     `sift_content_hashes` the new files' hashes, then it is saved as the
     current version, which fills the readings of a file below version 15 from
     the new `.sift` files;
   - an `embedded_patches` `.sfmr`: `workspace.contents` becomes the converted
     record, then it is saved as the current version;
   - a `.matches` file: the same three fields as a `sift_files` `.sfmr`, then
     it is written as the current version; a file of version 3 or older takes
     each image's width and height from its new `.sift` file, and a pairwise
     file of version 1 is written like any other, since the reader turns its
     two-view poses to the canonical convention on load;
   - a `.camrig` file below version 2: read and written as version 2. It names
     no features folder;
   - a `.kdf` index built from a version 1 record of the tree: deleted. It
     records the old folder and its hashes, and would read as `stale` once
     every `.sift` content hash changed, so deleting it loses nothing. The
     viewer reads a missing index beside a `.sfmr` as `none` and offers *Build
     Index Files*
     ([sift-index.md § none, current, stale](../../gui/sift-index.md#none-current-stale));
     the `<stem>-cluster-patches.matches` beside it is rewritten like any other
     `.matches` file and reads as stale until the index is built again
     ([index-files.md § When the cluster patches are stale](../../gui/index-files.md#when-the-cluster-patches-are-stale)).

   A `.sfmr` or `.matches` file below the current version that names no old
   folder is rewritten the same way with its record and hashes unchanged. A
   file already at the current version and naming no old folder is left as it
   is, so its hashes do not change.
5. **Write the workspace files.** Each version 1 workspace file of the tree is
   written as version 2 with every other option and field kept, hand-edited
   ones included:
   - **`sfmtool`**: `feature_options.gray_formula` becomes
     `feature_options.image_to_gray: {"formula": <gray_formula>}`, with no
     `quantize_bits`, since a version 1 `sfmtool` extraction did not round;
     options without a `gray_formula` gain the tool's default,
     `{"formula": "0.2126*R + 0.7152*G + 0.0722*B"}`;
   - **`colmap`** and **`opencv`**: `feature_options` gain the tool's
     conversion, `{"formula": "0.2126*R + 0.7152*G + 0.0722*B",
     "quantize_bits": 8}` for `colmap` and `{"formula": "0.299*R + 0.587*G +
     0.114*B", "quantize_bits": 8}` for `opencv`;
   - `feature_prefix_dir` becomes the new folder of phase 2,
     `features/{feature_type}-{hash}` over the converted options.

   When any file failed, in phase 1 or phase 4, no workspace file is written:
   each is left at version 1 and listed as not converted, and the report says
   why. Left at version 1, a workspace file still names its old folder on the
   next run, so the failed files are converted into the same new folder as the
   rest, even where the folder's own `.sift` files would name another (a
   workspace file of KerryPark360 records `feature_type: "sift-colmap"`, where
   the tool's workspace `feature_type` is `sift-colmap-max3000`).
6. **Delete the old folders.** Only when no file failed, every version 1
   features folder found in phase 1 is deleted, orphans included, except those
   it keeps ([What is kept](#what-is-kept)): its version 1 `.sift` files, then
   each directory that leaves empty, up to but not including the image
   directory. An orphan's image directory is not known, so the orphan folder
   and, when it is inside a `features` folder, that folder are removed when
   empty. A file in the folder that is not a version 1 `.sift` file is kept,
   keeps the folder, and is listed in the report. Phase 6 decides over every
   file of the whole tree, so a folder is deleted only when no file anywhere in
   the tree still needs it. When a file failed, nothing is deleted, and the
   report says to fix the failed files, or move them out of the tree, and run
   again.

The named image lists file and every file the walk does not read are left as
they are.

### What is not converted

These files are listed in the report and left unchanged. None of them stops
the deletion of the old folders:

- a cluster `.matches` file below version 6, which the reader refuses;
- any other `.sfmr` or `.matches` file the current reader refuses, such as a
  `.sfmr` whose camera lacks a parameter its model gained later. The old
  folders such a file names are kept, so its features stay with it should a
  later reader accept it; the other old folders are still deleted;
- a `.sfmr` or `.matches` file a `.sift` file of which is not on disk, cannot
  be read as the version 1 file it records, or is on disk but has changed; the
  reason says which;
- a `.sfmr` or `.matches` file whose `workspace` record resolves to a
  workspace outside the tree, which that workspace's upgrade converts;
- a `.sfmr` or `.matches` file of the tree an image name of which leaves the
  tree's root through `..`, which the current reader refuses, since the
  image-name rule refuses a name with `..`;
- a `.kdf` file that cannot be read.

A file the file system will not give up, or whose metadata cannot be read at
all, is not one of these: it **fails**, and stops phase 5 and the deletion,
since its record is unknown and its folder could be taken for an orphan.

### What is kept

Phase 6 keeps, and the report lists with the reason (`kept_because`):

- a folder that a file the current reader refuses names ("named by N file(s)
  the current reader refuses"), since that file's features are there;
- a folder that a file of a workspace outside the tree names, since the run
  cannot see that workspace's other files;
- a folder that a file of the tree names when another of its image names
  leaves the tree's root through `..`, since that file is not converted;
- a folder whose real path is not in the tree.

Phase 6 deletes nothing from a directory whose real path is outside the tree's
root, and when a folder it would delete lies outside the tree (reached through
a link) it deletes nothing at all and says so. When the tree holds any
symbolic link or junction to a directory, phase 6 deletes nothing either,
since the walk does not examine the files behind a link and one of them may
name an old folder; the report lists each link and says the old folders were
kept for that reason, and conversion and rewriting go ahead.

A folder that is not converted and is kept for one of these reasons is
**settled**: a rerun neither converts it nor counts it as work, so a workspace
that holds such files still reaches "nothing to do".

### A run that stops part way

A run stopped in phase 4, by Ctrl+C or a failure, leaves some files naming the
new folders, the rest naming the old ones, and the workspace files at
version 1. So does a run in which a file failed, since phase 5 then writes no
workspace file. Running the command again resumes it: the version 1 workspace
files name the same old folders, so they map to the same new folders, whose
converted files are kept, and the files still naming the old folders are
converted into them. No other command needs to handle that state: every
command that works with the workspace's settings or its `.sift` features
refuses a version 1 workspace, and a command that opens a file it is given
reads the `.sift` files that file records, which are on disk either way.

A run stopped after phase 5, or part way through phase 6, none of whose files
failed, leaves the workspace files at version 2, no file naming an old folder
but those listed as not converted, and some or all of the old folders on disk.
Running the command again finds them by the walk. A folder that no file names
any more is an orphan, whose files the first run converted, so the second run
deletes it without converting it; a folder still named by a file listed as not
converted is not converted again either. Until then they are folders no
converted file reads.

## Output

`sfm ws upgrade` prints, in order: the back-up line (not on `--dry-run`, where
it says instead that nothing was written); the tree's root; each workspace
file and its outcome; for each other kind of file, how many were converted,
already current, not converted, failed or deleted; every file not converted
and every file that failed with its reason, files that share a reason grouped
under it; each old folder, its new folder or "not converted", its `.sift` file
count, and whether it is deleted or kept and why, with any files the deletion
left in it; the number of `.sift` files converted; the linked directories the
walk did not enter; the report's notes; and "Nothing to do" when there was
nothing to do. Paths in the tree are relative to its root. While the run goes,
a line every ten seconds says how far its current step has got.

The command exits with status 1 when any file failed, after printing the
report, and when the run is refused or stopped.

A version 1 workspace with eight reconstructions and two match sets:

```
$ sfm ws upgrade SeoulBull --dry-run
Dry run: nothing was written. Each outcome below is what the upgrade would do.
Workspace tree: /home/user/SeoulBull
Workspace files:
  .sfm-workspace.json: converted to version 2
Files:
  .sfmr: 8 converted
  .matches: 2 converted
Old features folders:
  images/features/sift-sfmtool-3dcd2b2f8c892d12c3ffe28cedce19c9 -> images/features/sift-sfmtool-9017ec0d2c5dfc401da610578062b53d (17 .sift files); deleted
Converted .sift files: 17
Notes:
  Saving a .sfmr file changes its content_xxh128, so a point ID (pt3d_<hash>_<index>) written down from a rewritten .sfmr before the upgrade no longer resolves.

$ sfm ws upgrade SeoulBull
Back up the workspace before upgrading it: this rewrites .sfmr, .matches and .camrig files in place and deletes the old features folders, which cannot be undone.
Workspace tree: /home/user/SeoulBull
...

$ sfm ws upgrade SeoulBull
Back up the workspace before upgrading it: this rewrites .sfmr, .matches and .camrig files in place and deletes the old features folders, which cannot be undone.
Workspace tree: /home/user/SeoulBull
Workspace files:
  .sfm-workspace.json: already current
Files:
  .sfmr: 8 already current
  .matches: 2 already current
Nothing to do: no file is left to convert and no old features folder to delete.
```

A copy of a rig workspace, some of whose `.sfmr` files the current reader
refuses, which keeps the old folders they name, and some of whose `.sfmr`
files record the workspace they were copied from, outside this tree:

```
$ sfm ws upgrade KerryPark480-copy
Back up the workspace before upgrading it: ...
Workspace tree: /home/user/KerryPark480-copy
Workspace files:
  .sfm-workspace.json: converted to version 2
Files:
  .sfmr: 53 converted, 32 not converted
  .matches: 6 converted
  .kdf: 1 already current
Not converted:
  27 files: it records the workspace /data/KerryPark480, which is outside this tree, so that workspace's upgrade converts it
    sfmr/bugbash-20260929/kerry.sfmr
    ...
  5 files: the current reader refuses it, so it is left as it is: Invalid format: invalid camera intrinsics: missing parameter 'bspline_coeff_count' for camera model 'SFMTOOL_FISHEYE'
    sfmr/seed-hypotheses/groundworld-20260817-h0-release.sfmr
    ...
Old features folders:
  fisheye_left/features/sift-sfmtool-3dcd2b2f8c892d12c3ffe28cedce19c9 -> fisheye_left/features/sift-sfmtool-9017ec0d2c5dfc401da610578062b53d (24 .sift files); kept: named by 5 file(s) the current reader refuses
  fisheye_right/features/sift-sfmtool-3dcd2b2f8c892d12c3ffe28cedce19c9 -> fisheye_right/features/sift-sfmtool-9017ec0d2c5dfc401da610578062b53d (24 .sift files); kept: named by 5 file(s) the current reader refuses
Converted .sift files: 48
Notes:
  Saving a .sfmr file changes its content_xxh128, ...
```

Run again, it reports each kept folder as not converted and kept, and says
"Nothing to do".

## Errors

Each of these stops the run with exit status 1. All but the last two are found
in phases 1 and 2, before anything is written, so `--dry-run` gives them too.

| Case | What it says |
|------|--------------|
| No `.sfm-workspace.json` in `PATH` or above it | no workspace file was found |
| `PATH` is in a workspace nested in another (`NotTreeRoot`) | names the outermost workspace and says to run the upgrade there, which upgrades every workspace nested in it |
| A workspace file that cannot be read, or is not version 1 or 2 | names the file and what is wrong |
| A directory of the tree cannot be listed | names it |
| A `.sift` file the file system will not give up (`UnreadableSiftFiles`) | lists each with its error, and says to close the program that holds it or make a cloud copy available offline, then run again |
| A version 1 record that cannot be converted (`FeatureOptions`) | why, and every file that holds it: for an `sfmtool-undistort` record, to run `sfm undistort` again, since its `.sift` files are made from its source's; a tool other than `sfmtool`, `colmap` or `opencv`; a `gray_formula` that does not parse |
| A `feature_prefix_dir` that is empty, absolute or holds `..` (`FeaturePrefixDir`) | the prefix and every file that holds it |
| A folder whose `.sift` files record different tools or options (`FolderFeatureOptions`) | the folder and two files that differ, and to move the files that do not belong elsewhere |
| A file that names folders that convert to different new folders (`FolderTargets`) | the file and the folders |
| Two old folders hold different files for one new file (`Conflicts`) | each pair and the file both would become, and to remove the one that is not wanted |
| Version 1 `.sift` files that cannot be converted (`SiftFiles`) | every one with its reason (an image missing, unreadable or changed, or another file already in the new folder), and that no other file was rewritten and nothing deleted; new `.sift` files may have been written |
| Ctrl+C | that the upgrade was stopped, and to run it again to finish |

A file that fails in phase 1 or phase 4 does not stop the run: the report
lists it, and the command exits with status 1 after printing it.

## Interface

The command is `upgrade` in
[`_commands/ws.py`](../../../src/sfmtool/_commands/ws.py), which calls the
binding `sfmtool.workspace.upgrade_workspace` with a `ProgressCounter` and
prints the report it returns through `report_lines`. The work is one Rust
function,
[`sfmtool_core::workspace::upgrade_workspace`](../../../crates/sfmtool-core/src/workspace/upgrade.rs),
which plans and writes in one call:

```rust
pub fn upgrade_workspace(
    path: &Path,                       // a directory or file in the tree's root workspace
    options: &UpgradeWorkspaceOptions, // { dry_run: bool }
    progress: &Progress<'_>,
) -> Result<UpgradeWorkspaceReport, UpgradeWorkspaceError>
```

```rust
use sfmtool_core::workspace::{upgrade_workspace, UpgradeWorkspaceOptions};

let report = upgrade_workspace(path, &UpgradeWorkspaceOptions { dry_run: true }, &progress)?;
if report.nothing_to_do() {
    println!("nothing to do");
}
```

It takes a `&Progress`, since it runs for minutes on a large workspace: it
opens the phases `find files`, `plan`, `convert .sift files`, `rewrite files`,
`workspace file` and `delete old folders`, each counting what it works
through, and checks for cancellation between files; a cancelled run is one of
the stopped runs above.

`UpgradeWorkspaceReport` holds the tree's root (`workspace_dir`); the
workspaces of the tree (`workspaces`, the root first); the dry-run flag; one
`UpgradedFile` per file (each workspace file of the tree first, in the order
of `workspaces`, then each `.sfmr`, `.matches`, `.camrig` and `.kdf` file the
walk found) with its `UpgradeFileKind` and `UpgradeOutcome`: `Converted`,
`AlreadyCurrent`, `NotConverted(reason)` (one of
[What is not converted](#what-is-not-converted), which does not stop the
deletion), `Failed(reason)` (could not be read or written, which does) or
`Deleted`; one `UpgradedFolder` per version 1 folder, with its new folder
(`None` when it is not converted), its `.sift` file count, why it is kept
(`kept_because`, `None` when phase 6 deletes it), whether it was deleted and
the files the deletion kept, and `is_settled`; the number of `.sift` files
converted; the linked directories the walk did not enter; whether phase 6
deleted every folder it does not keep (`old_folders_deleted`); and notes for
the user, the point-ID note among them. `nothing_to_do` holds when every
workspace file is current, no file is converted or deleted, and every folder
is settled. A dry run fills the same report with the outcomes the run would
have. The report serializes with `serde`.

`UpgradeWorkspaceError` is what stops a run, each case in [Errors](#errors):
`NoWorkspace`, `NotTreeRoot`, `WorkspaceFile`, `ListFolder`,
`UnreadableSiftFiles`, `FeatureOptions`, `FeaturePrefixDir`,
`FolderFeatureOptions`, `FolderTargets`, `Conflicts`, `SiftFiles` and
`Cancelled`. Every one but `SiftFiles` and `Cancelled` is raised before
anything but new `.sift` files is written.

The binding,
[`sfmtool.workspace.upgrade_workspace(path, *, dry_run=False, progress=None)`](../../../crates/sfmtool-py/src/workspace.rs),
returns the report as a dict, with `nothing_to_do` added as a key, and raises
`FileNotFoundError` for no workspace, `OSError` for a directory that cannot be
listed, the exception a signal handler raised while it ran
(`KeyboardInterrupt` for Ctrl+C), `InterruptedError` for a run cancelled
otherwise, and `ValueError` for the rest. The
`ProgressCounter` holds the count of the phase that is running.

The `.kdf` files are read by `sfmtool_kdf_format::read_kdf_workspace`, which
reads a file's recorded workspace from its metadata alone, since phase 1 needs
only the folder each index was built from.

The `.sfmr` rewrite is in Rust: `SfmrReconstruction::load`, replace
`metadata.workspace.contents` and the two hash columns of
`ObservationSource::SiftFiles`, then `save`. Python's `SfmrReconstruction` has
no way to change a loaded reconstruction's `workspace.contents`, and the dict
route, `read_sfmr` then `write_sfmr`, applies no version 4 to version 5
convention upgrade and refuses a file below version 15 until its readings are
supplied. `load` fills a `sift_files` file's inline keypoints from the old
`.sift` files when the file has none, so it runs while the old folders are
still on disk. `save` gathers the readings of a file below version 15 from the
value's own rows, then each image's verified version 2 `.sift` file, then its
photograph; the switch to the new folder happens before the save, so the
gather reads them from the new `.sift` files rather than decoding every
photograph.

The conversion of a record's options is `upgrade_feature_options` and the new
folder's name `feature_prefix_dir`, both in
[`sfmtool_core::workspace`](../../../crates/sfmtool-core/src/workspace/feature_options.rs)
and bound in `sfmtool.workspace`, as `sfm ws init` names its folder; the
viewer's *Build SIFT Files* makes the same conversion in memory for a version
1 record ([index-files.md](../../gui/index-files.md#building-the-sift-files)).
