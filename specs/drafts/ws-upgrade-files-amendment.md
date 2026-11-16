# `sfm ws upgrade` brings the workspace's files to the current versions

**Status:** Draft. Decided: `sfm ws upgrade` itself does the whole upgrade (no
new command); `.sift` files are converted, never extracted again; `.sfmr`,
`.matches` and `.camrig` files are rewritten at their own paths; the old
version 1 features folders are deleted at the end of a run that converted
everything, with no option to keep them, and the command and its spec tell the
user to back up the workspace first. Targets sfmtool 0.4.1. Open questions are
at the end.

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

## The command

```bash
sfm ws upgrade [PATH] [--dry-run]
```

The workspace is found as today. On a version 1 workspace the command runs the
phases below in order. On a version 2 workspace it runs every phase that still
has work, phase 6 included, so a run that stopped part way can be run again to
finish, and a version 2 workspace that holds older files is brought up to
date. When no file needs converting and no version 1 features folder remains,
it prints "nothing to do" and changes nothing.

`--dry-run` runs phases 1 and 2 and prints the plan (each old folder and its
new one, how many files of each kind would be converted, and every file that
cannot be, with the reason) and writes nothing.

1. **Find the files.** Walk the workspace for `.sfmr`, `.matches`, `.camrig`
   and `.kdf` files and for the `.sift` folders named by a `feature_prefix_dir`
   — the workspace file's own and each one recorded in a `.sfmr` or `.matches`
   file's `workspace.contents`. A `feature_prefix_dir` is relative to each
   image's directory, so one record names one folder in every directory that
   holds images
   ([workspace.md § Feature Storage Convention](../workspace/workspace.md#feature-storage-convention)).
   A workspace can hold several records' folders, left by earlier
   `sfm ws init --force` runs. The walk also finds every version 1 features
   folder whether or not a record names it: each directory in the workspace
   that holds `.sift` files is checked with
   `sfmtool_core::features::sift::version_1_sift_files`, which lists the
   version 1 `.sift` files directly in one folder, the same check every
   writer of version 2 files makes before it writes into a folder. A version
   1 folder that no `.sfmr` or `.matches` file names and that is not
   the workspace file's own folder is an **orphan**: it is not converted, it
   is deleted in phase 6 with the others, and the report lists it.
2. **Map each old folder to a new one.** Each distinct version 1 record
   (`feature_tool`, `feature_type`, `feature_options`, `feature_prefix_dir`)
   is converted with `upgrade_feature_options`, and its new folder is named
   from the converted options by the one folder-hash implementation (see
   "Steps" for why there must be one). The workspace file's own record maps
   to the folder phase 5 writes into it.
3. **Convert the `.sift` files** of every old folder that is not an orphan
   into its new folder, as above, in parallel. This builds a table from each old file's
   `content_xxh128` to the new file's `content_xxh128` and
   `feature_tool_xxh128`. An image whose file is missing, cannot be decoded or
   no longer matches `image_file_xxh128` cannot be converted; its `.sift` is
   listed and the run fails at the end of this phase, before any other file is
   rewritten.
4. **Rewrite the files that record the old folders, and the `.camrig`
   files,** each atomically at its own path:
   - a `sift_files` `.sfmr`: `workspace.contents` becomes the converted record
     and `feature_tool_hashes` and `sift_content_hashes` are looked up in the
     table, then it is saved as the current version, which fills the readings
     of a file below version 15 from the new `.sift` files;
   - an `embedded_patches` `.sfmr`: `workspace.contents` becomes the converted
     record, then it is saved as the current version;
   - a `.matches` file: the same three fields as a `sift_files` `.sfmr`, then
     it is written as the current version; a file of version 3 or older takes
     each image's width and height from its new `.sift` file;
   - a `.camrig` file: read and written as the current version. It records no
     features folder; it is rewritten only so that no version 1 `.camrig` is
     left for a reader that drops version 1;
   - a `.kdf` index: deleted, since it records the old folder and its hashes.
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
   A file already at the current version and naming no old folder is left as
   it is.
5. **Write the workspace file** as version 2, as `sfm ws upgrade` does today,
   with the new folder from phase 2.
6. **Delete the old folders.** Only when phases 3 to 5 converted every file
   they had to, every version 1 features folder found in phase 1 is deleted.
   Otherwise nothing is deleted, and the report says what to fix before
   running the command again.

### A run that stops part way

A run stopped in phase 4 leaves some files naming the new folders, the rest
naming the old ones, and the workspace file at version 1. Running the command
again resumes it. No other command needs to handle that state: every command
that works with the workspace's settings or its `.sift` features refuses a
version 1 workspace and says to run `sfm ws upgrade`
([workspace.md § A command that uses a workspace requires version 2](../workspace/workspace.md#a-command-that-uses-a-workspace-requires-version-2)),
and a command that opens a file it is given reads the `.sift` files that file
records, which are on disk either way, since phase 3 has finished and phase 6
has not run.

A run stopped after phase 5, or part way through phase 6, leaves the workspace
file at version 2, no file naming an old folder, and some or all of the old
folders on disk. Running the command again finds them by the walk in phase 1,
as orphans, since no file names them any more and they are not the workspace
file's own folder. Their files were converted by the first run, so the second
run converts nothing and runs phase 6, which deletes them. Until then they
are folders no file reads: commands use the new folders through the version 2
workspace file and the rewritten files' records.

### What is not converted

These files are listed in the report and left unchanged. None of them stops
the old folders from being deleted, because none of them can be read through
those folders by the current reader anyway:

- a cluster `.matches` file below version 6, which the reader refuses;
- a `.sfmr` or `.matches` file whose recorded `.sift` files are already
  missing or changed, so it no longer matches any file in an old folder.

### Backing up

The command deletes the old features folders and rewrites `.sfmr` and
`.matches` files in place, so it cannot be undone. Its help, its spec and the
first line it prints before converting anything say to back up the workspace
first. Saving changes each `.sfmr` file's `content_xxh128`, so a point ID
(`pt3d_<hash>_<index>`) written down before the upgrade no longer resolves;
the report says so.

## Interface

The work is one function in `sfmtool-core`'s `workspace` module, which takes a
`&Progress` (it runs for minutes on a large workspace) and returns a report
listing every file and its outcome; it plans and writes in one call, with a
dry-run flag. It is bound in `sfmtool.workspace`, and the `upgrade` command in
[`_commands/ws.py`](../../src/sfmtool/_commands/ws.py) calls it and prints the
report. `upgrade_workspace` in `_workspace.py` becomes the workspace file step
of that function, so the Python wrapper does not repeat the binding's work.

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
3. **The workspace upgrade above,** with this draft converted into
   [ws-upgrade-command.md](../cli/workspace/ws-upgrade-command.md) and
   [workspace.md § Migration](../workspace/workspace.md#migration), and tests on
   copies of `test-data` workspaces, which are version 1. The datasets carry
   no `.sift` or `.matches` files, so the tests write version 1 `.sift` files
   into the copies first.

## Open questions

- **`sfmtool-undistort` records.** `upgrade_feature_options` refuses them.
  Whether to convert an undistorted workspace's `.sift` files with their
  recorded conversion, or to keep refusing and say to run `sfm undistort`
  again.
- **Version 1 `.matches` files.** Whether a pairwise file of version 1 is
  written back as version 10 or listed as not converted. The reader already
  turns its two-view poses to the canonical convention on load and reports the
  value as version 10, so writing it back is the same write as for any other
  file of version 3 or older.
