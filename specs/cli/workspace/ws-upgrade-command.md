# `sfm ws upgrade` Command

`sfm ws upgrade` moves a version 1 workspace to version 2. It rewrites the
workspace's `.sfm-workspace.json` as version 2, keeping every feature option,
including hand-edited ones, and names a new features folder from the converted
options, so the workspace's `.sift` files are extracted again, as version 2
files, into that folder. Nothing else on disk changes: the old folder and its
`.sift` files, the named image lists file and every `.sfmr` stay as they are,
and a `.sfmr` that names the old folder still finds its `.sift` files through
its own record. The two versions of the file, and why extraction refuses
version 1 settings, are in
[workspace.md § Versions](../../workspace/workspace.md#versions).

## Command Syntax

```bash
sfm ws upgrade [PATH]
```

The workspace is found by the usual upward search for `.sfm-workspace.json`
from `PATH`, or from the current directory when `PATH` is omitted, so the
command works from any directory inside the workspace.

## What it does

On a version 1 file:

- **`sfmtool`**: `feature_options.gray_formula` becomes
  `feature_options.image_to_gray: {"formula": <gray_formula>}`, with no
  `quantize_bits`, since a version 1 `sfmtool` extraction did not round. Options
  without a `gray_formula` gain the tool's default,
  `{"formula": "0.2126*R + 0.7152*G + 0.0722*B"}`.
- **`colmap`** and **`opencv`**: `feature_options` gain the tool's conversion,
  `{"formula": "0.2126*R + 0.7152*G + 0.0722*B", "quantize_bits": 8}` for
  `colmap` and `{"formula": "0.299*R + 0.587*G + 0.114*B", "quantize_bits": 8}`
  for `opencv`.

Every other option and field is kept, `version` becomes `2`, and
`feature_prefix_dir` becomes `features/{feature_type}-{hash}`, the hash taken
over the converted options as `sfm ws init` takes it. The command prints the
file, the conversion and the old and new folders.

On a version 2 file it prints "nothing to do: already version 2" and changes
nothing.

## Errors

Each error names the file and changes nothing:

| Case | What it says |
|------|--------------|
| No `.sfm-workspace.json` in `PATH` or above it | no workspace file was found |
| The file has no `version` | it has no `version` |
| A `version` other than 1 or 2 | the version found |
| `feature_tool` is `sfmtool-undistort` | to run `sfm undistort` again, since the `.sift` files of an undistorted workspace are made from its source's and cannot be extracted again |
| Any other `feature_tool` than `sfmtool`, `colmap` or `opencv` | the tool is not one of the three |
| A `gray_formula` that does not parse | that the `gray_formula` does not parse, and why |

## Example

```bash
$ sfm ws upgrade my_project
Upgraded workspace file: /home/user/my_project/.sfm-workspace.json
  version: 1 -> 2
  feature_tool: sfmtool
  image_to_gray: 0.2126*R + 0.7152*G + 0.0722*B
  old features folder: features/sift-sfmtool-3dcd2b2f8c892d12c3ffe28cedce19c9
  new features folder: features/sift-sfmtool-9017ec0d2c5dfc401da610578062b53d
The old folder and its .sift files are left on disk; .sift files are extracted again into the new folder.

$ sfm ws upgrade my_project
/home/user/my_project/.sfm-workspace.json: nothing to do: already version 2
```

## Interface

The command is `upgrade` in
[`_commands/ws.py`](../../../src/sfmtool/_commands/ws.py), which calls
`upgrade_workspace` in [`_workspace.py`](../../../src/sfmtool/_workspace.py).
The conversion of the options is `upgrade_feature_options` in
[`sfmtool_core::workspace`](../../../crates/sfmtool-core/src/workspace/feature_options.rs),
bound as `sfmtool.workspace.upgrade_feature_options`; the viewer's *Build SIFT
Files* makes the same conversion in memory for a version 1 record
([index-files.md](../../gui/index-files.md#building-the-sift-files)). The new
folder name comes from `get_feature_tool_xxh128` in
[`sift/file.py`](../../../src/sfmtool/sift/file.py), as `sfm ws init`'s does.
