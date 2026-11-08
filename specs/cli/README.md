# CLI Specifications

One spec per `sfm` subcommand, in the directory matching the category the
command is registered under in `src/sfmtool/cli.py` — the same grouping
`sfm --help` prints. Implementations live in `src/sfmtool/_commands/`.

## How `sfm` loads its commands

`sfm` imports a command's module only when that command is looked up: when it
runs, or when its own `--help` is printed. `COMMANDS` in
[cli.py](../../src/sfmtool/cli.py) lists every top-level command but `version`
with its name, its `--help` category, the module in `sfmtool._commands` and the
attribute that define it, and its one-line help, the first sentence of the
command's docstring. `CategoryGroup` in
[_cli_group.py](../../src/sfmtool/_cli_group.py) lists the commands by
category from that table, so `sfm --help` imports no command module, and
`sfm explorer` does not import the modules of the commands that need numpy,
OpenCV or pycolmap. The package root works the same way:
[`sfmtool/__init__.py`](../../src/sfmtool/__init__.py) binds the extension's
three root-level names on import, which takes about 10 ms, and the names of its
Python submodules and the binding modules such as `sfmtool.fileio` on first use,
through a module `__getattr__`. For an unknown
command name `CategoryGroup` suggests the close matches among every command
name (`sfm solv` gives "Did you mean 'solve'?"), not only among the commands
loaded so far.

So a command module does no work at import time beyond defining its command,
and `sfmtool._commands/__init__.py` imports none of them. A new command is a
row in `COMMANDS`, with the first sentence of its docstring as the one-line
help; [test_lazy_loading.py](../../tests/test_lazy_loading.py) checks that each
row names a command defined where the row says, under that name, with that
help and category, that every command defined in a `sfmtool._commands`
module has a row, and that `import sfmtool` and `sfm --help` import none of
numpy, OpenCV, pycolmap and the command modules.

A helper that several commands share is a `_`-named module in
`sfmtool._commands`, so it has no row in `COMMANDS`.
[`_range_options.py`](../../src/sfmtool/_commands/_range_options.py) holds
the `-r/--range` and `--named-image-list` options that ten commands share
([below](#the-shared--r--range-option)), and the `--filter-points` option of
`sfm to-colmap-bin` and `sfm to-nerfstudio`.
[`_sfmr_path.py`](../../src/sfmtool/_commands/_sfmr_path.py) provides
`check_sfmr_path`, which raises a `click.UsageError` reading "<label> must be
a .sfmr file, got: <path>" when a path argument does not end in `.sfmr`
(compared case-insensitively), so a wrong argument fails as a usage error
rather than inside a reader or writer.
`sfm align`, `analyze`, `compare`, `densify`, `embed-patches`, `epipolar`,
`flow` (for its optional reconstruction), `from-colmap-bin`, `heatmap`,
`merge`, `panorama`, `render-patches`, `to-colmap-bin`, `to-nerfstudio`,
`undistort`, `web-export` and `xform` call it on their `.sfmr` arguments, so
they all report that mistake in the same words.

## The shared `-r/--range` option

Ten commands take `-r/--range`:

- On image directories, matched against each file's number by
  `number_from_filename`: `sfm sift`, `sfm match`, `sfm motion`,
  `sfm pick-images` and `sfm solve`.
- On a reconstruction's images: `sfm analyze` (with `--metrics`),
  `sfm panorama`, `sfm to-colmap-bin` and `sfm to-nerfstudio`.
- On a `.matches` file's images: `sfm estimate-intrinsics`, as `sfm solve`
  does when given a `.matches` file.

Its value is either a range expression of file numbers or the name of one of
the workspace's named image lists
([`../workspace/named-image-lists.md`](../workspace/named-image-lists.md)),
optionally followed by a slice. Each of the ten commands handles it the same
way:

- **A value with a letter or a `[` in it is a list name; any other value is a
  range.** A range expression contains only digits, `-`, `:` and `,` (and
  whitespace), so the two cannot be confused. A value with a letter that is
  not a valid name (`walk 1`, say) is an error saying what a name may contain.
  A range given alone reaches the command as before, so each command's
  handling of a range is its own.
- **A slice after the name takes part of the list by position**, with Python's
  list slice syntax and rules: `-r 'walk1[:30]'` is the list's first 30
  images, `-r 'walk1[-30:]'` its last 30 and `-r 'walk1[::3]'` every third
  one. An item is one image file, not one frame, counted from 0 in the list's
  order among the images the command would take from the list without a
  slice, and a slice never looks at frame numbers. Negative indexes and a
  negative step count from the end, and a bound beyond either end is clamped
  to it. A single index (`walk1[5]`) and a step of 0 are errors, as is a
  malformed slice or one with no valid name before it, before any file is
  read; a slice that keeps no image is an error naming the lists file, the
  list, the slice and how many images there were. `[` and `]` are glob
  characters in bash and zsh, so a value with a slice is quoted. The rules in
  full are in
  [named-image-lists.md § Slices](../workspace/named-image-lists.md#slices).
- **`--named-image-list NAME`** is the long form for a list, for scripts that
  want to say which they mean. Its value must be a valid name, and may carry a
  slice (`--named-image-list 'walk1[:30]'`).
- **One name and one range may be given together**, as two `-r` options or as
  `--named-image-list` with one `-r`. The images are then the list's members
  whose frame number the range also selects: the range narrows every entry of
  the list, by the number the entry's frame field captures. An entry with no
  frame field has no frame number to narrow, so narrowing a list with such an
  entry is an error naming the entry, and a range that leaves no members is
  an error naming the list and the range. A slice given with a range indexes
  the narrowed list: `-r 'walk1[:30]' -r 100-900` is the first 30 of the
  members whose frame number is in 100-900. Two names, or two ranges, are an
  error.
- **An unknown name** is an error that names the lists file and lists the
  names it defines, or says the workspace has no lists file, and says: "A
  --range value with a letter in it is read as a named image list; a range
  contains only digits, '-', ':' and ','." A list name given where no
  workspace is found is an error saying that no workspace was found and so no
  named image lists are defined, with the same sentence.

**The image-directory commands.** For `sift`, `match`, `motion`, `pick-images`
and `solve`, a list replaces the image path arguments: the command takes either
paths (or, for `motion`, a `.sfmr` file) or a list, and giving both is an
error. `solve` given a single `.matches` file takes a list as a filter
instead, described below. The workspace is the one the upward search from
the current directory finds, and the images are the list's members, in the
list's order, as absolute paths. Only the members that are image files are
used, by the same extension test the command's own directory expansion applies
(`.png`, `.jpg` and `.jpeg` for `sift`, `match`, `pick-images` and `solve`;
those and `.tif`, `.tiff` and `.bmp` for `motion`; compared without regard to
case), so a pattern such as `images/**` does not hand a command a `.sift` or
`.json` file. The others are skipped without a message, as a directory listing
skips them, and a list none of whose members is an image file is an error
naming the list. A slice indexes the image files that test leaves, and the
command takes them in the slice's order. `match`, `pick-images` and `solve` use
the list's workspace as theirs.

**`solve` with a `.matches` file.** A list given with a single `.matches` file
keeps the file's images that are among the list's images, and the pairs
between them; no matching runs again. The list is resolved in the workspace
the `.matches` file resolves to, with the extension test, the range and the
slice applied as for image paths, so a slice counts positions in the list,
not among the file's images. The images of the list the file does not hold
are ignored, and the command prints how many; when the file holds none of
them, it is an error naming the lists file, the list and the `.matches`
file. The kept images stay in the file's order. A range given alone keeps the
file's images whose file number it holds. `estimate-intrinsics` selects the
images of its `.matches` file the same way, through the same function, and
prints the count of the list's images the file does not hold on stderr, so
its `--json` output stays parseable
([estimate-intrinsics-command.md](reconstruction/estimate-intrinsics-command.md#selecting-images)).

**The reconstruction commands.** For `analyze --metrics`, `panorama`,
`to-colmap-bin` and `to-nerfstudio`, the list is resolved in the
reconstruction's workspace (the one its `.sfmr` resolves to on load),
matched against the reconstruction's image names rather than the filesystem,
so it works where the image files are absent. It selects the reconstruction's
images that are members, in the reconstruction's order; members the
reconstruction does not hold are ignored, and it is an error, naming the list,
when none are left. A slice indexes the members the reconstruction holds, in
the list's order, and the images it keeps are then used in the
reconstruction's order. `--filter-points` keeps its meaning, and accepts a list in
place of a range. `analyze --metrics` reports which list it used and how many
images it kept, and takes its reconstruction-wide figures and flag thresholds
over every image, as it does for a range.

A `.sfmr` that `sfm solve` writes from a list records nothing about the list,
as it records nothing about a range.

```bash
sfm solve -i -r walk1_sharp            # the list's images
sfm sift --extract -r rig_sharp -r 10-20   # the list, narrowed to frames 10-20
sfm solve -i -r 'walk1_sharp[:30]'     # the list's first 30 images
sfm solve -g pairs.matches -r 'walk1_sharp[:30]'  # the same 30, from the pairs file
sfm to-colmap-bin result.sfmr out/ --named-image-list hero_shots
```

## Writing a `.sfmr`

Every command that writes a `.sfmr` (`solve`, `xform`, `merge`, `align`,
`densify`, `embed-patches`, `from-colmap-bin`, `undistort`, and SfM
Explorer's Save) writes the current format version, 16, which, like version 15, stores every
image's sharpness and gradient structure tensor
([sfmr-file-format.md § Image readings](../formats/sfmr-file-format.md#image-readings-version-15)).
A command whose input already holds them carries them with the images. One
that writes from `.sift` files copies them from the version 2 files, and an
input older than version 15 has them gathered when it is saved: from each
image's `.sift` file, and otherwise from the image itself, decoded and checked
against the reconstruction's record of its hash
([§ Where a writer gets the values](../formats/sfmr-file-format.md#where-a-writer-gets-the-values)).
So saving an older `.sfmr` needs its `.sift` files or its images. When an
image has neither, the command fails before it writes anything, with an error
naming each such image, the `.sift` path tried and the image path tried.

## Workspace

| Command | Spec |
|---------|------|
| `sfm ws init` | [ws-init-command.md](workspace/ws-init-command.md) |
| `sfm ws upgrade` | [ws-upgrade-command.md](workspace/ws-upgrade-command.md) |
| `sfm camrig` (`create`, `cp`, `spherical-tiles`) | [camrig-command.md](workspace/camrig-command.md) |
| `sfm pano2rig` | [pano2rig-command.md](workspace/pano2rig-command.md) |
| `sfm insv2rig` | [insv2rig-command.md](workspace/insv2rig-command.md) |

## Image Feature

| Command | Spec |
|---------|------|
| `sfm sift` | [sift-command.md](image-feature/sift-command.md) |
| `sfm match` | [match-command.md](image-feature/match-command.md) |
| `sfm pick-images` | [pick-images-command.md](image-feature/pick-images-command.md) |
| `sfm cluster-patches` | [cluster-patches-command.md](image-feature/cluster-patches-command.md) |

## Reconstruction

| Command | Spec |
|---------|------|
| `sfm solve` | [solve-command.md](reconstruction/solve-command.md) |
| `sfm inspect` | [inspect-command.md](reconstruction/inspect-command.md) |
| `sfm analyze` | [analyze-command.md](reconstruction/analyze-command.md) |
| `sfm compare` | [compare-command.md](reconstruction/compare-command.md) |
| `sfm align` | [align-command.md](reconstruction/align-command.md) |
| `sfm merge` | [merge-command.md](reconstruction/merge-command.md) |
| `sfm densify` | [densify-command.md](reconstruction/densify-command.md) |
| `sfm motion` | [motion-command.md](reconstruction/motion-command.md) |
| `sfm embed-patches` | [embed-patches-command.md](reconstruction/embed-patches-command.md) |
| `sfm estimate-intrinsics` | [estimate-intrinsics-command.md](reconstruction/estimate-intrinsics-command.md) |
| `sfm xform` | [xform/](reconstruction/xform/) — see below |

### `sfm xform` sub-commands

| Sub-command | Spec |
|-------------|------|
| the command and its shared transforms | [xform-command.md](reconstruction/xform/xform-command.md) |
| `--refine-normals` | [refine-normals-command.md](reconstruction/xform/refine-normals-command.md) |
| `--refine-keypoints` | [refine-keypoints-command.md](reconstruction/xform/refine-keypoints-command.md) |
| `--localize-keypoints` | [localize-keypoints-command.md](reconstruction/xform/localize-keypoints-command.md) |
| `--include-by-distribution` | [select-by-distribution-command.md](reconstruction/xform/select-by-distribution-command.md) |
| `--find-points-at-infinity` | [find-points-at-infinity.md](reconstruction/xform/find-points-at-infinity.md) |
| `--scale-by-measurements` | [scale-by-measurements-command.md](reconstruction/xform/scale-by-measurements-command.md) |

## Visualization

| Command | Spec |
|---------|------|
| `sfm epipolar` | [epipolar-command.md](visualization/epipolar-command.md) |
| `sfm explorer` | [explorer-command.md](visualization/explorer-command.md) — launches the viewer specced under [../gui/](../gui/README.md) |
| `sfm heatmap` | [heatmap-command.md](visualization/heatmap-command.md) |
| `sfm render-patches` | [render-patches-command.md](visualization/render-patches-command.md) |
| `sfm panorama` | [panorama-command.md](visualization/panorama-command.md) |
| `sfm web-export` | [web-export-command.md](visualization/web-export-command.md) |

## Automation

Commands that let a script or agent drive a running program.

| Command | Spec |
|---------|------|
| `sfm explorer-call` | [explorer-call-command.md](automation/explorer-call-command.md) — calls one tool on a running SfM Explorer's MCP endpoint |

## Image Processing

| Command | Spec |
|---------|------|
| `sfm flow` | [flow-command.md](image-processing/flow-command.md) |
| `sfm undistort` | [undistort-command.md](image-processing/undistort-command.md) |

## COLMAP Interop

The COLMAP file reading and writing these commands share is described in
[../formats/colmap-interop.md](../formats/colmap-interop.md).

| Command | Spec |
|---------|------|
| `sfm to-colmap-bin` | [to-colmap-bin-command.md](colmap-interop/to-colmap-bin-command.md) |
| `sfm to-colmap-db` | [to-colmap-db-command.md](colmap-interop/to-colmap-db-command.md) |
| `sfm from-colmap-bin` | [from-colmap-bin-command.md](colmap-interop/from-colmap-bin-command.md) |
| `sfm to-nerfstudio` | [to-nerfstudio-command.md](colmap-interop/to-nerfstudio-command.md) |

## Other

`sfm version` is the one command not in `COMMANDS`; `sfm --help` lists it
under this heading.

| Command | Spec |
|---------|------|
| `sfm version` | [version-command.md](other/version-command.md) |
