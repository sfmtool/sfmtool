# `sfm ws upgrade` brings the workspace's files to the current versions

**Status:** Draft. Decided: `sfm ws upgrade` itself does the whole upgrade (no
new command); `.sift` files are converted, never extracted again; `.sfmr`,
`.matches` and `.camrig` files are rewritten at their own paths; the old
version 1 features folders are deleted at the end of a run that converted
everything, with no option to keep them, and the command and its spec tell the
user to back up the workspace first. Targets sfmtool 0.4.1. The upgrade itself
is built ([Interface](#interface)); `sfm ws upgrade` does not call it yet, which
is the second half of step 3 ([Steps](#steps)). The questions this draft left
open are settled at the end.

Amends [`../cli/workspace/ws-upgrade-command.md`](../cli/workspace/ws-upgrade-command.md)
and [`../workspace/workspace.md` § Migration](../workspace/workspace.md#migration),
which describe the shipped command and point back here.

`sfm ws upgrade` today rewrites only `.sfm-workspace.json`
([ws-upgrade-command.md](../cli/workspace/ws-upgrade-command.md)): the
workspace's `.sift` files are extracted again into a new folder, and every
existing `.sfmr` and `.matches` file keeps naming the old folder of version 1
`.sift` files. Readers are to drop the format versions older than those 0.4
writes (`.sfmr` 16, `.matches` 10, `.sift` 2, `.camrig` 2). Once they do, a
workspace upgraded the way it is today loses every reconstruction and match set
made before the upgrade, because their features are in files nothing reads.
This draft makes `sfm ws upgrade` convert those files instead, so a workspace
made by sfmtool 0.3 or a development build comes out of one command with every
file at the current version and pointing at version 2 `.sift` files.

## Why the `.sift` files are converted, not extracted again

Extracting again does not reproduce the version 1 features. For the `sfmtool`
tool, #1143 moved the grey conversion from `f64` to `f32` after version 2 was
introduced; on its own measurement 0.04% of keypoints appear on one side only
and 2.8% of descriptors differ. For `colmap`, the result depends on the
pycolmap build and the GPU. A `.sfmr` or `.matches` file records feature
indexes into its `.sift` files, so new features would invalidate every one of
them.

A version 1 file converts to version 2 losslessly: positions, affine shapes,
descriptors and the thumbnail are copied byte for byte; `feature_options`
becomes `upgrade_feature_options(feature_tool, feature_options)`; the
sharpness and gradient structure tensor are computed from the image file
through the converted options' `image_to_gray`, after the image is verified
against the file's `image_file_xxh128`; `metadata.version` becomes 2. Both of
the file's hashes change: `feature_tool_xxh128` is the hash of
`feature_tool_metadata.json`, whose `feature_options` change, and
`content_xxh128` covers that entry and `metadata.json`, which gains the version
and the two readings, as well as the feature arrays and the thumbnail
([sift-file-format.md](../formats/sift-file-format.md)). That is why every file
that records them has to be rewritten too.

No version 1 record carries `double_image_below_long_side`, and
`upgrade_feature_options` adds none, so an upgraded workspace keeps doubling
every image as its features were extracted (the standard version 1 sfmtool
options convert to `features/sift-sfmtool-9017ec0d2c5dfc401da610578062b53d`);
adopting the 4096-pixel limit `sfm ws init` now records takes `sfm ws init
--force` and extraction into a new folder.

## The command

```bash
sfm ws upgrade [PATH] [--dry-run]
```

The workspace is found as today, and the run upgrades the whole **tree** of
workspaces it belongs to: the outermost workspace at or above it, the tree's
root, and every workspace nested in that one. One run does the whole tree
because a file of one workspace can name the `.sift` files of another, in
either direction, and two runs that each decide what to delete without seeing
the other's files can each delete what the other still needs. When the path
given is in a workspace nested in another, the run refuses before reading
anything else, names the outermost workspace and says to run the command
there. The command then runs the phases below in order. On a tree whose
workspace files are all version 2 it runs every phase that still has work,
phase 6 included, so a run that stopped part way can be run again to finish,
and a version 2 workspace that holds older files is brought up to date. When
no file needs converting and no version 1 features folder remains, it prints
"nothing to do" and changes nothing.

`--dry-run` runs phases 1 and 2 and prints the plan (each old folder and its
new one, each orphan folder, which would be deleted without converting, how
many files of each kind would be converted, and every file that cannot be,
with the reason) and writes nothing.

1. **Find the files.** Walk the tree for `.sfmr`, `.matches`, `.camrig`
   and `.kdf` files and for the `.sift` folders named by a `feature_prefix_dir`
   — each workspace file's own and each one recorded in a `.sfmr` or `.matches`
   file's `workspace.contents`. A `feature_prefix_dir` is relative to each
   image's directory, so one record names one folder in every directory that
   holds images
   ([workspace.md § Feature Storage Convention](../workspace/workspace.md#feature-storage-convention)).
   A workspace can hold several records' folders, left by earlier
   `sfm ws init --force` runs. The walk also finds every version 1 features
   folder whether or not a record names it: each directory in the tree
   that holds `.sift` files is checked by the rule
   `sfmtool_core::features::sift::version_1_sift_files` applies, which
   lists the version 1 `.sift` files directly in one folder, the same check
   every writer of version 2 files makes before it writes into a folder. The
   upgrade reads them with `scan_sift_folder`, which also reports each
   `.sift` file whose metadata could not be read: one the file system will
   not give up (locked, or a sync client's placeholder that cannot be
   fetched) stops the run before anything is written, listing each such
   file and its error, since it could hold features a file needs; one whose
   content the reader rejects is passed over, as the writers pass it over.
   A version 1 folder that no file and no workspace file of the tree names is
   an **orphan**: it is not converted, it is deleted in phase 6 with the
   others, and the report lists it.

   The walk enters nested workspaces and lists them, but not a symbolic link
   or junction to a directory, whose target may lie outside the tree; the
   report lists each. Every file is attributed to the workspace its
   `workspace` record resolves to (by the rule every reader resolves it
   with), and its image names and folders are relative to that workspace. A
   `.sfmr` or `.matches` file whose record resolves to a workspace outside the
   tree is left as it is and listed as not converted, and a folder inside the
   tree that such a file names is kept in phase 6, since the run cannot see
   that workspace's other files.

   The folders a file's record names are found by their paths on disk,
   `<its workspace>/<image's directory>/<feature_prefix_dir>` for each image
   the file records, with each `..` of an image name resolved by name (a
   file of a nested workspace can name `../images/<image>`), from the file's
   record and image names alone, read by
   the cheap readers (`read_sfmr_image_names`, `read_matches_image_names`)
   rather than a whole load, so a file the current reader refuses still
   names its folders, wherever its record resolves. A file of the tree an
   image name of which leaves the tree's root through `..` is not converted,
   since the run does not convert the folder outside the tree it names, and
   the folders it names in the tree are kept in phase 6. A file whose
   image names cannot be read names the folders its record's
   `feature_prefix_dir` names below its workspace. A file whose metadata or
   image names the file system will not give up fails, wherever its record
   would resolve, and stops the deletion. Each `.sfmr` and `.matches` file the run would rewrite
   is read whole here, in a dry run too: a file the current reader refuses is
   told apart from one the file system will not give up (it is locked, or
   cannot be opened or read), which is a failure, by the error's type, and a
   `.sift` file a file records that is not on disk, or not the version 1 file
   it records, is found before anything is written.
2. **Map each old folder to a new one.** A version 1 folder is named by a
   file when it is one of the folders phase 1 found for it, and by a
   version 1 workspace file's record (`feature_tool`, `feature_type`,
   `feature_options`, `feature_prefix_dir`) when that workspace is the
   deepest of the tree the folder lies in, the folder is
   `<image dir>/<feature_prefix_dir>` below it, and, when files name the
   folder, `<image dir>` is the directory of their images. So a workspace
   file names no folder of a workspace nested in its own, and a prefix that
   ends another (`sift-x` beside `features/sift-x`) is not taken for it
   where files show which one the folder is. Several records
   can name one folder: a workspace file of sfmtool 0.3 records
   `feature_type: "sift-sfmtool"` and its `.sfmr` files `"sift"`, or two
   files record options that differ. Each physical old folder therefore maps
   to exactly one new folder, `<image dir>/<new feature_prefix_dir>`, and one
   converted record, named by the one folder-hash implementation (see
   "Steps" for why there must be one):
   - when a version 1 workspace file's own record names the folder, the
     record that
     workspace file is converted to, with `upgrade_feature_options`, which
     is what phase 5 writes into it, as `sfm ws upgrade` names it today;
   - otherwise the folder's own `.sift` files decide: the record of their
     feature tool, the tool's workspace `feature_type`
     (`sfmtool_core::workspace::feature_type_for_tool`, the `sift-<tool>`
     form `sfm ws init` records, bound as
     `sfmtool.workspace.feature_type_for_tool` and the one implementation
     behind Python's `get_feature_type_for_tool`), and their
     `feature_options` converted. When the folder's `.sift` files record
     different tools or options, the run stops here, naming the folder and
     two files that differ.

   Every file that names the folder, whatever its record, is pointed at
   that one new folder and takes that one converted record, so the records
   come out the same. A file that names folders that map to different new
   folders stops the run here, naming the file and the folders. A file that
   names no folder on disk takes its workspace file's converted record when
   that names the same `feature_prefix_dir`, and otherwise its own record
   converted under the tool's workspace `feature_type`. A record that cannot be converted
   (an `sfmtool-undistort` record, a tool this build does not know, a
   `gray_formula` that does not parse) stops the run here, naming the files
   that hold it, since its folder could be neither converted nor deleted. So
   does a record whose `feature_prefix_dir` is empty, absolute or holds `..`,
   whose folder would not lie below each image's directory, and a version 1
   workspace file whose options already carry `image_to_gray`, which
   `upgrade_feature_options` refuses. Two old folders whose records convert
   to the same new folder, such as a
   hand-named `features/sift-colmap-3000` beside a hashed one with the same
   options, are converted into it together: a file the two hold with the
   same content hash is converted once, and two different files for one
   image stop the run here, naming both, rather than the run picking one.
3. **Convert the `.sift` files** of every old folder that a version 1
   workspace file or a file this run rewrites names, into its new folder, as
   above, in parallel; a folder named only by files the run leaves is not
   converted, since none of them would read the new files. The image of
   `<image dir>/<feature_prefix_dir>/<name>.sift` is `<image dir>/<name>`,
   never the file's `image_name`, which version 1 extractors sometimes wrote
   relative to another directory. This builds a table from each old file's
   `content_xxh128` to the new file's `content_xxh128` and
   `feature_tool_xxh128`. An image whose file is missing, cannot be decoded or
   no longer matches `image_file_xxh128` cannot be converted; its `.sift` is
   listed and the run fails at the end of this phase, before any other file is
   rewritten. A new file already in the new folder is kept when it is this
   file's conversion (the same tool, type, image and features, with the
   options upgraded), as a rerun finds it, and is otherwise refused the same
   way rather than replaced, since it may be an extraction other files
   record.
4. **Rewrite the files that record the old folders, and the `.camrig`
   files,** each atomically at its own path (every writer below writes a
   temporary file beside the target and renames it over it). A `.sfmr` or
   `.matches` file below the current version that names no old folder is
   rewritten the same way, with its record and hashes unchanged:
   - a `sift_files` `.sfmr`: `workspace.contents` becomes the converted record
     of the folders it names (phase 2)
     and `feature_tool_hashes` and `sift_content_hashes` are looked up in the
     table, then it is saved as the current version, which fills the readings
     of a file below version 15 from the new `.sift` files;
   - an `embedded_patches` `.sfmr`: `workspace.contents` becomes the converted
     record, then it is saved as the current version;
   - a `.matches` file: the same three fields as a `sift_files` `.sfmr`, then
     it is written as the current version; a file of version 3 or older takes
     each image's width and height from its new `.sift` file. A pairwise file
     of version 1 is written back as version 10 like any other: the reader
     turns its two-view poses to the canonical convention on load;
   - a `.camrig` file: read and written as the current version. It records no
     features folder; it is rewritten only so that no version 1 `.camrig` is
     left for a reader that drops version 1;
   - a `.kdf` index whose sources record a version 1 record of this
     workspace: deleted, since it records the old folder and its hashes.
     No reader rebuilds a missing `.kdf` on its own. The viewer is the one
     reader that finds an index by itself, `<stem>-sift-index.kdf` beside a
     `.sfmr`: it reads a missing one as `none` and one whose recorded `.sift`
     content hashes are not those of the files the node resolves as `stale`,
     refuses a bench search in both states, and offers *Build Index Files* or
     *Rebuild Index Files*
     ([sift-index.md § none, current, stale](../gui/sift-index.md#none-current-stale)).
     The bindings open only a path their caller names. A kept index would read
     as `stale` after the upgrade, since every `.sift` content hash changes,
     so deleting it loses nothing. The `<stem>-cluster-patches.matches` beside
     it is rewritten like any other `.matches` file, and reads as stale in the
     viewer until the index is built again, since its index is gone
     ([index-files.md § When the cluster patches are stale](../gui/index-files.md#when-the-cluster-patches-are-stale)).
   A file already at the current version and naming no old folder (a `.sfmr`
   at version 16, a `.matches` at version 10, a `.camrig` at version 2) is
   left as it is, so its hashes do not change.
5. **Write the workspace files**: each version 1 workspace file of the tree
   as version 2, as `sfm ws upgrade` does today for one, with the new folder
   from phase 2, keeping every other key. When any file failed, in phase 1
   or phase 4, no workspace file is written: each is left at version 1 and
   listed as not converted, and the report says why. Phase 2 maps a folder a
   version 1 workspace file names to that workspace file's new folder, which
   can differ from the one the folder's own `.sift` files would give (a
   workspace file of KerryPark360 records `feature_type: "sift-colmap"`,
   where the tool's workspace `feature_type` is `sift-colmap-max3000`); left
   at version 1, the workspace file still names the folder on the next run,
   so the failed files are converted into the same new folder as the rest.
6. **Delete the old folders.** Only when phases 3 to 5 converted every file
   they had to, every version 1 features folder found in phase 1 is deleted:
   its version 1 `.sift` files, then each directory that leaves empty, up to
   but not including the image directory. An orphan's image directory is not
   known, so the orphan folder and, when it is inside a `features` folder,
   that folder are removed when empty. A file in the folder that is not a version
   1 `.sift` file is kept, keeps the folder, and is listed in the report.
   Phase 6 decides over every file of the whole tree, so a folder is deleted
   only when no file anywhere in the tree still needs it. Four kinds of
   folder are kept, and listed with the reason: a folder that a file the
   current reader refuses names ("kept: named by N files the current reader
   refuses"), since that file's features are there; a folder that a file of
   a workspace outside the tree names, since the run cannot see that
   workspace's other files; a folder that a file of the tree names when
   another of its image names leaves the tree's root through `..`, since
   that file is not converted; and a folder whose real path is not in the
   tree.
   Phase 6 deletes nothing from a directory whose real path is outside the
   tree's root, and when a folder it would delete lies outside the tree
   (reached through a link), it deletes nothing at all and says so. When the
   tree holds any symbolic link or junction to a directory, phase 6 deletes
   nothing either: the walk does not examine the files behind a link, and one
   of them may name an old folder. The report lists each link and says the
   old folders were kept for that reason; conversion and rewriting go ahead.
   Otherwise nothing is deleted, and the report says what to fix before
   running the command again: a file that could not be read or rewritten
   stops the deletion, while a file that is not converted for a reason under
   "What is not converted" does not.

### A run that stops part way

A run stopped in phase 4 leaves some files naming the new folders, the rest
naming the old ones, and the workspace files at version 1. So does a run in
which a file failed, since phase 5 then writes no workspace file. Running the
command again resumes it: the version 1 workspace files name the same old
folders, so they map to the same new folders, whose converted files are kept,
and the files still naming the old folders are converted into them. No other command needs to handle that state: every command
that works with the workspace's settings or its `.sift` features refuses a
version 1 workspace and says to run `sfm ws upgrade`
([workspace.md § A command that uses a workspace requires version 2](../workspace/workspace.md#a-command-that-uses-a-workspace-requires-version-2)),
and a command that opens a file it is given reads the `.sift` files that file
records, which are on disk either way, since phase 3 has finished and phase 6
has not run.

A run stopped after phase 5, or part way through phase 6, none of whose files
failed, leaves the workspace files at version 2, no file naming an old folder but those listed under "What
is not converted", and some or all of the old folders on disk. Running the
command again finds them by the walk in phase 1. A folder that no file names
any more is an orphan, since it is no workspace file's own folder either;
its files were converted by the first run, so the second run does not convert
them. A folder still named by a file listed as not converted is not an orphan,
but no file the run rewrites names it, so it is not converted again either.
Phase 6 then deletes them all, except the folders it keeps. Until then they
are folders no converted file reads: commands use the new folders through the
version 2 workspace files and the rewritten files' records. A folder kept
because a file the reader refuses names it is settled: a rerun neither
converts it nor counts it as work, so a workspace that holds such files still
reaches "nothing to do".

### What is not converted

These files are listed in the report and left unchanged. None of them stops
the deletion of the old folders, because none of them can be read through
those folders by the current reader anyway:

- a cluster `.matches` file below version 6, which the reader refuses;
- any other `.sfmr` or `.matches` file the current reader refuses, such as a
  `.sfmr` whose camera lacks a parameter its model gained later (KerryPark480
  holds five whose `SFMTOOL_FISHEYE` camera has no `bspline_coeff_count`).
  The old folders such a file names are kept, not deleted, so its features
  stay with it should a later reader accept it; the other old folders are
  still deleted. A file the file system will not give up (locked, or it
  cannot be opened or read) is not one of these: it fails, and stops the
  deletion; so does a file whose metadata cannot be read at all, since its
  record is unknown and its folder could be taken for an orphan;
- a `.sfmr` or `.matches` file a `.sift` file of which is not on disk, cannot
  be read as the version 1 file it records (its content is not one the
  reader accepts), or is on disk but has changed; the reason says which;
- a `.sfmr` or `.matches` file whose `workspace` record resolves to a
  workspace outside the tree, which that workspace's upgrade converts; a
  folder inside the tree that it names is kept;
- a `.sfmr` or `.matches` file of the tree an image name of which leaves
  the tree's root through `..`; a folder inside the tree that it names is
  kept;
- a `.kdf` file that cannot be read.

### Backing up

The command deletes the old features folders and rewrites `.sfmr` and
`.matches` files in place, so it cannot be undone. Its help, its spec and the
first line it prints before converting anything say to back up the workspace
first. Saving changes each `.sfmr` file's `content_xxh128`, so a point ID
(`pt3d_<hash>_<index>`) written down before the upgrade no longer resolves;
the report says so.

## Interface

The work is one function,
[`sfmtool_core::workspace::upgrade_workspace`](../../crates/sfmtool-core/src/workspace/upgrade.rs),
which plans and writes in one call. It takes a `&Progress`, since it runs for
minutes on a large workspace, and a settings struct whose one field is the
dry-run flag; it returns a report listing every file and folder and its
outcome:

```rust
pub fn upgrade_workspace(
    path: &Path,                       // a directory or file in the workspace
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

`UpgradeWorkspaceReport` holds the tree's root (`workspace_dir`); the
workspaces of the tree (`workspaces`, the root first); the dry-run flag; one
`UpgradedFile` per file (each workspace file of the tree first, in the order
of `workspaces`, then each `.sfmr`,
`.matches`, `.camrig` and `.kdf` file the walk found) with its
`UpgradeFileKind` and `UpgradeOutcome`: `Converted`, `AlreadyCurrent`,
`NotConverted(reason)` (one of "What is not converted", which does not stop the
deletion), `Failed(reason)` (could not be read or written, which does) or
`Deleted`; one `UpgradedFolder` per version 1 folder, with its new folder
(`None` when it is not converted: an orphan, or a folder only files the run
leaves name), its `.sift` file count, why it is kept (`kept_because`, `None`
when phase 6 deletes it), whether it was deleted and the files the deletion
kept; a folder not converted and kept is settled (`is_settled`); the number of
`.sift` files converted; the linked directories the
walk did not enter; whether phase 6 deleted every folder it does not keep;
and notes for the user, the point-ID note among them. `nothing_to_do` holds
when every workspace file is current, no file is converted or deleted, and
every folder is settled. A dry run fills the same report with the outcomes the
run would have. The report serializes with `serde`, so a caller can print or
store it as JSON.

`UpgradeWorkspaceError` is what stops a run before anything but new `.sift`
files is written: no workspace, a path in a workspace nested in another
(`NotTreeRoot`, naming the outermost workspace, where to run it), a
workspace file that cannot be read or is not version 1 or 2, a directory that
cannot be listed, a `.sift` file in a folder the run examines that the file
system will not give up (`UnreadableSiftFiles`, each with its error), a
folder whose `.sift` files record different tools or options
(`FolderFeatureOptions`), a file whose folders map to different new folders
(`FolderTargets`), a
record that
cannot be converted (`FeatureOptions`, naming the files that hold it), a
`feature_prefix_dir` that would leave the image directory
(`FeaturePrefixDir`), two old files for one new one (`Conflicts`), every
`.sift` file that could not be converted (`SiftFiles`), and `Cancelled`. The
progress phases are `find files`, `plan`,
`convert .sift files`, `rewrite files`, `workspace file` and
`delete old folders`, each counting what it works through; cancellation is
checked between files, and a cancelled run is one of the stopped runs above.

It is bound as `sfmtool.workspace.upgrade_workspace(path, *, dry_run=False,
progress=None)`, which returns the report as a dict and raises
`FileNotFoundError` for no workspace, `OSError` for a directory that cannot be
listed, `InterruptedError` for a signal and `ValueError` for the rest. The
`upgrade` command in [`_commands/ws.py`](../../src/sfmtool/_commands/ws.py)
is to call it and print the report, and `upgrade_workspace` in
`_workspace.py`, whose workspace file step the Rust function now holds, is to
be removed, so the Python wrapper does not repeat the binding's work.

The `.kdf` files are read by `sfmtool_kdf_format::read_kdf_workspace`, which
reads a file's recorded workspace from its metadata alone, with no scalar type
and no tree or row map, since phase 1 needs only the folder each index was
built from.

The `.sfmr` rewrite is in Rust. Python's `SfmrReconstruction` has no way to
change a loaded reconstruction's `workspace.contents`
(`clone_with_changes` takes the two hash columns but no workspace record), and
the dict route that could, `read_sfmr` then `write_sfmr`, applies no
version 4 to version 5 convention upgrade and refuses a file below version 15
until its readings are supplied. In Rust it is `SfmrReconstruction::load`,
replace `metadata.workspace.contents` and the two hash columns of
`ObservationSource::SiftFiles`, then `save`. `load` also fills a `sift_files`
file's inline keypoints from the old `.sift` files when the file has none, so
it runs while the old folders are still on disk. `save` gathers the readings
of a file below version 15 from the value's own rows, then each image's
verified version 2 `.sift` file, then its photograph; the switch to the new
folder happens before the save, so the gather reads them from the new `.sift`
files rather than decoding every photograph.

## Steps

Each step is its own pull request.

1. **One folder-hash implementation.** `get_feature_tool_xxh128` in
   `sift/file.py` and `feature_prefix_dir` in
   `sfmtool_core::workspace` compute the same name separately, and their float
   formatting can differ. Phase 2 runs in Rust, so Rust's must be the one, with
   Python calling it, and must give the names Python gives today for the
   existing workspaces. The 2026-11-13 hygiene audit lists the duplication,
   in "Have the image-pair-graph bindings take the reconstruction, and stop
   repeating bindings in Python" under
   [Findings from the mechanical checks](../../reports/2026-11-13-hygiene-audit.md#findings-from-the-mechanical-checks).
2. **`.sift` version 1 to version 2 conversion** in core, with a binding, and
   tests that a converted file's features are byte-identical and its readings
   equal a fresh computation from the image.
3. **The workspace upgrade above,** in two pull requests:
   - **3a.** `upgrade_workspace` in core and its binding, with tests on
     synthetic workspaces built from `test-data` images, which write version 1
     `.sift`, `.sfmr`, `.matches`, `.camrig` and `.kdf` files first, since the
     datasets carry none.
   - **3b.** `sfm ws upgrade` calls the binding and prints the report;
     `upgrade_workspace` in `_workspace.py` is removed; and this draft is
     converted into
     [ws-upgrade-command.md](../cli/workspace/ws-upgrade-command.md) and
     [workspace.md § Migration](../workspace/workspace.md#migration) and
     retired.

## Settled questions

- **`sfmtool-undistort` records** are refused, as step 2 decided: their
  `.sift` files are made from their source's, and the message says to run
  `sfm undistort` again. A workspace whose files record one stops in phase 2,
  before anything is written.
- **A version 1 pairwise `.matches` file** is written back as version 10. The
  reader already turns its two-view poses to the canonical convention on load,
  so writing it back is the same write as for any other file of version 3 or
  older.
