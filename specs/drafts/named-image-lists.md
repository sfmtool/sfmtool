# Named image lists

**Status:** Draft. Decided by the maintainer: a workspace names sets of its
images in its own file, `.sfm-named-image-lists.json`, in the workspace root
beside `.sfm-workspace.json`; no `.sfmr` references a list; `-r/--range`
accepts either a range expression or a list's name, with `--named-image-list`
as the long form. The choices this draft makes inside those decisions are
listed in [Choices made here](#choices-made-here) for review. The lists file,
its reader and its resolver are implemented and specified in
[`../workspace/named-image-lists.md`](../workspace/named-image-lists.md); the
command-line side below is not implemented.

A capture often holds more images than one solve should use: the sharp frames
of a walk, the frames of one rig run, a few hand-picked photographs. This draft
lets a workspace give such a set a name, defined by image patterns and frame
ranges in a small file beside the workspace file, so that a command can be
pointed at the set by name instead of by paths and ranges repeated on every
command line.

It amends these standing specs, each of which links back here:

- [`../workspace/named-image-lists.md`](../workspace/named-image-lists.md):
  the lists file and its resolver, which the commands below call.
- [`../cli/README.md`](../cli/README.md): the `-r/--range` option that eight
  commands share.

The command that will write named image lists, a later `sfm survey`-style
command that reads the sharpness and gradient structure tensor of a capture's
frames ([image-sharpness.md](image-sharpness.md)) and proposes lists of the
sharp ones, is the intended producer; it is not designed here.

## The lists file

The lists file, its validation, the order of a list's members, narrowing and
the resolver the commands below call are specified in the standing spec
[`../workspace/named-image-lists.md`](../workspace/named-image-lists.md), which
links back here for the command-line side. What this draft adds is how the
commands reach a list.

**Finding it.** A command first finds the workspace by the existing workspace
discovery, then reads the lists file beside that workspace's
`.sfm-workspace.json`. A command that has no workspace has no lists, so a list
name given to it is an error saying that no workspace was found and so no named
image lists are defined. If the workspace has no lists file, it has no named
image lists, so giving a list name is an error that says so.

## The CLI's `-r/--range`

Eight commands take `-r/--range` today:

- On image directories, matched against each file's number by
  `number_from_filename`: `sfm sift`, `sfm match`, `sfm motion` and `sfm solve`.
- On a reconstruction's images: `sfm analyze` (with `--metrics`),
  `sfm panorama`, `sfm to-colmap-bin` and `sfm to-nerfstudio`.

Each of them gains the same handling:

- **A value with a letter in it is a list name; any other value is a range.** A
  range expression contains only digits, `-`, `:` and `,` (and whitespace), so
  the two cannot be confused. A value with a letter that is not a valid name
  (`walk 1`, say) is an error saying what a name may contain.
- **`--named-image-list NAME`** is the long form for a list, for scripts that
  want to say which they mean. Its value must be a valid name.
- **One name and one range may be given together**, as two `-r` options or as
  `--named-image-list` with one `-r`. The images are the list's members whose
  frame number the range also selects: the range narrows every entry of the
  list. An entry with no frame field has no frame number to narrow, so narrowing
  a list with such an entry is an error naming the entry. Two names, or two
  ranges, are an error.
- **An unknown name** is an error that names the lists file (or says the
  workspace has none), lists the names it does define, and says: "A --range value with a letter in it is read as a
  named image list; a range contains only digits, '-', ':' and ','."

**Which workspace, and which images.** For the four commands on image
directories, a list replaces the image path arguments: the command takes either
paths or a list, and giving both is an error. The workspace is the one found by
the upward search from the current directory, and the images are the list's
members, in the list's order. For the four commands on a reconstruction, the
list is resolved in the reconstruction's workspace, matched against the
reconstruction's image names rather than the filesystem (so it works where the
image files are absent), and selects the reconstruction's images that are
members, in the reconstruction's order; members the reconstruction does not hold
are ignored, and it is an error, naming the list, when none are left. `--filter-points` keeps its meaning.

A `.sfmr` that `sfm solve` writes from a list records nothing about the list,
as it records nothing about a range today.

## Implementation

- **The named lists** resolve through `sfmtool.workspace.NamedImageLists`
  ([`../workspace/named-image-lists.md`](../workspace/named-image-lists.md)):
  `resolve` for the four commands on image directories and
  `resolve_in_names` for the four on a reconstruction, with `narrow` for a
  name given with a range. The `-r` handling moves into one shared option in
  `src/sfmtool/_commands/_range_options.py` that all eight commands use.

## Choices made here

Decisions inside the maintainer's decisions that this draft makes and that
review should check:

1. **With a list, the image-directory commands take no image paths** and find
   the workspace from the current directory; narrowing a list that has an entry
   with no frame field is an error.

## Not in this draft

The command that writes named image lists, and any rule for which frames go in
one: it will be designed separately.
