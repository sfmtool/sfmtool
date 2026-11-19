# `sfm ws upgrade` Command

`sfm ws upgrade` brings the files of a version 2 workspace to the current file
versions. It rewrites each `.sfmr`, `.matches` and `.camrig` file below the
current version, that the current reader still reads, at its own path as the
current version, so that a later release that stops reading those versions
reads every file in the workspace. It rewrites files in place, so it cannot be
undone: **back up the workspace first** ([Backing up](#backing-up)).

A workspace sfmtool 0.3 or a development build made holds a version 1
`.sfm-workspace.json` and `.sift`, `.sfmr`, `.matches` and `.camrig` files of
versions this release does not read. The command refuses such a workspace
before it reads anything else: sfmtool 0.4.1 is the last release that reads
those versions, and its `sfm ws upgrade` converts the workspace to version 2.
Every other command that uses a workspace refuses a version 1 one the same way
([workspace.md § A command that uses a workspace requires version 2](../../workspace/workspace.md#a-command-that-uses-a-workspace-requires-version-2)).
The versions of the workspace file are in
[workspace.md § Versions](../../workspace/workspace.md#versions), and those of
each file format in
[format-versions.md](../../formats/format-versions.md).

## Command Syntax

```bash
sfm ws upgrade [PATH] [--dry-run]
```

The workspace is found by the usual upward search for `.sfm-workspace.json`
from `PATH`, or from the current directory when `PATH` is omitted. The run
upgrades the **tree** of that workspace: the workspace itself, the tree's
root, and every workspace nested in it, so that one run brings every file
below the root to the current versions. A workspace that encloses the root is
not part of the tree.

`--dry-run` finds the files and decides what to do with each (phases 1 and 2
below), prints the same report a run would, and writes nothing.

When no file needs rewriting, the command says "Nothing to do" and changes
nothing.

## Backing up

The command rewrites `.sfmr`, `.matches` and `.camrig` files in place, so it
cannot be undone. Its help, this spec, and the first line it prints before
rewriting anything say to back up the workspace first. Saving changes each
`.sfmr` file's `content_xxh128`, so a point ID (`pt3d_<hash>_<index>`) written
down from a `.sfmr` before the upgrade no longer resolves; the report says so
when it rewrote a `.sfmr`.

## What it does

Before anything else, the command reads the root's workspace file and refuses
it when it is version 1, or cannot be read, or is of a version other than 2.

The command then runs these phases in order.

1. **Find the files.** It walks the tree for `.sfmr`, `.matches` and
   `.camrig` files, entering nested workspaces. It reads each nested
   workspace's file and refuses the run, before reading any other file, when
   one is version 1, cannot be read or is of a version other than 2. It does
   not enter a symbolic link or junction to a directory, whose target may lie
   outside the tree or lead back into it; the report lists each one.
2. **Decide what to do with each file.** It reads each file's metadata, and a
   file at the current version is left as it is, so its hashes do not change.
   A file below the current version is read whole, as phase 3 will read it: a
   `.sfmr` by `SfmrReconstruction::load`, which checks what the format reader
   leaves to it, such as each camera's parameters, and finds the workspace the
   file records; a `.matches` or `.camrig` file by its format reader. This
   happens in a dry run too, so the report tells a file the run rewrites from
   one the current reader refuses. A file the file system will not give up
   (locked, or it cannot be opened or read) **fails**; one whose content the
   current reader refuses is left as it is ([What is not
   converted](#what-is-not-converted)).
3. **Rewrite the files**, each atomically at its own path (a temporary file
   beside the target, renamed over it):
   - a `.sfmr` file is loaded and saved as the current version, which fills
     the image readings of a file below version 15 from the reconstruction's
     own rows, then each image's `.sift` file, then its photograph;
   - a `.matches` file is read and written as the current version; a file of
     version 3 or older takes each image's width and height from its `.sift`
     file, in the folder its record names, and a pairwise file of version 1 is
     written like any other, since the reader turns its two-view poses to the
     canonical convention on load;
   - a `.camrig` file below version 2 is read and written as version 2.

The workspace files, the `.sift` files, the `.kdf` indexes, the named image
lists file and every other file are left as they are: the readers read every
version of them that a version 2 workspace holds.

### What is not converted

These files are listed in the report with the reason, and left unchanged:

- a cluster `.matches` file below version 6, which the reader refuses; the
  report says to regenerate it with `sfm match --cluster`;
- any other `.sfmr`, `.matches` or `.camrig` file the current reader refuses,
  such as a `.sfmr` whose camera lacks a parameter its model gained later, one
  whose recorded workspace cannot be found, or a file that is not an archive.

A file the file system will not give up is not one of these: it **fails**,
and the command exits with status 1 after printing the report.

### A run that stops part way

A run stopped in phase 3, by Ctrl+C, leaves some files rewritten and the rest
as they were. Each file is rewritten atomically, so none is left half
written. Running the command again finishes it: the rewritten files are at the
current version and are left as they are.

## Output

`sfm ws upgrade` prints, in order: the back-up line (not on `--dry-run`, where
it says instead that nothing was written); the tree's root; the nested
workspaces, when there are any; for each kind of file, how many were
converted, already current, not converted or failed; every file not converted
and every file that failed with its reason, files that share a reason grouped
under it; the linked directories the walk did not enter; the report's notes;
and "Nothing to do" when there was nothing to do. Paths in the tree are
relative to its root. While the run goes, a line every ten seconds says how
far its current step has got.

The command exits with status 1 when any file failed, after printing the
report, and when the run is refused or stopped.

A version 2 workspace with a version 1 `.camrig` file and a current one:

```
$ sfm ws upgrade KerryPark --dry-run
Dry run: nothing was written. Each outcome below is what the upgrade would do.
Workspace tree: /home/user/KerryPark
Files:
  .camrig: 1 converted, 1 already current

$ sfm ws upgrade KerryPark
Back up the workspace before upgrading it: this rewrites .sfmr, .matches and .camrig files in place, which cannot be undone.
Workspace tree: /home/user/KerryPark
Files:
  .camrig: 1 converted, 1 already current

$ sfm ws upgrade KerryPark
Back up the workspace before upgrading it: ...
Workspace tree: /home/user/KerryPark
Files:
  .camrig: 2 already current
Nothing to do: no file is left to convert.
```

## Errors

Each of these stops the run with exit status 1. All but the last are found
before anything is written, so `--dry-run` gives them too.

| Case | What it says |
|------|--------------|
| No `.sfm-workspace.json` in `PATH` or above it | no workspace file was found |
| A workspace file of the tree is version 1 (`Version1Workspace`) | names the workspace file, says it is version 1 and that this sfmtool upgrades a version 2 workspace, and that sfmtool 0.4.1 is the last release that reads the older `.sift`, `.sfmr`, `.matches` and `.camrig` versions a version 1 workspace holds: install it, run `sfm ws upgrade` on the workspace, then use the workspace with this release |
| A workspace file that cannot be read, or is of a version other than 1 or 2 (`WorkspaceFile`) | names the file and what is wrong |
| A directory of the tree cannot be listed (`ListFolder`) | names it |
| Ctrl+C | that the upgrade was stopped, and to run it again to finish |

The refusal of a version 1 workspace reads, for example:

```
the workspace file /home/user/SeoulBull/.sfm-workspace.json is version 1, and
this sfmtool upgrades a version 2 workspace. sfmtool 0.4.1 is the last release
that reads the older .sift, .sfmr, .matches and .camrig versions a version 1
workspace holds: install it, run `sfm ws upgrade` on the workspace, then use
the workspace with this release.
```

A file that fails does not stop the run: the report lists it, and the command
exits with status 1 after printing it.

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
opens the phases `find files`, `plan` and `rewrite files`, each counting what
it works through, and checks for cancellation between files; a cancelled run
is the stopped run above.

`UpgradeWorkspaceReport` holds the tree's root (`workspace_dir`); the
workspaces of the tree (`workspaces`, the root first); the dry-run flag; one
`UpgradedFile` per `.sfmr`, `.matches` and `.camrig` file the walk found, in
that order of kinds and each kind sorted by path, with its `UpgradeFileKind`
and `UpgradeOutcome`: `Converted`, `AlreadyCurrent`, `NotConverted(reason)`
(one of [What is not converted](#what-is-not-converted)) or `Failed(reason)`
(could not be read or written); the linked directories the walk did not
enter; and notes for the user, the point-ID note among them. `nothing_to_do`
holds when no file is converted or failed. A dry run fills the same report
with the outcomes the run would have. The report serializes with `serde`.

`UpgradeWorkspaceError` is what stops a run, each case in [Errors](#errors):
`NoWorkspace`, `Version1Workspace`, `WorkspaceFile`, `ListFolder` and
`Cancelled`. The release the version 1 refusal names is
`WORKSPACE_LAST_RELEASE_READING_VERSION_1` in
[`sfmtool_core::workspace`](../../../crates/sfmtool-core/src/workspace/feature_options.rs).

The binding,
[`sfmtool.workspace.upgrade_workspace(path, *, dry_run=False, progress=None)`](../../../crates/sfmtool-py/src/workspace.rs),
returns the report as a dict, with `nothing_to_do` added as a key, and raises
`FileNotFoundError` for no workspace, `OSError` for a directory that cannot be
listed, the exception a signal handler raised while it ran
(`KeyboardInterrupt` for Ctrl+C), `InterruptedError` for a run cancelled
otherwise, and `ValueError` for a workspace file it refuses. The
`ProgressCounter` holds the count of the phase that is running.

The `.sfmr` rewrite is in Rust, `SfmrReconstruction::load` then `save`:
Python's dict route, `read_sfmr` then `write_sfmr`, applies no version 4 to
version 5 convention upgrade and refuses a file below version 15 until its
readings are supplied.
