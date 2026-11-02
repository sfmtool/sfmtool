# Named image lists

**Status:** Draft. Decided by the maintainer: a workspace names sets of its
images in its own file, `.sfm-named-image-lists.json`, in the workspace root
beside `.sfm-workspace.json`; no `.sfmr` references a list; `-r/--range`
accepts either a range expression or a list's name, with `--named-image-list`
as the long form. The choices this draft makes inside those decisions are
listed in [Choices made here](#choices-made-here) for review. Nothing here is
implemented.

A capture often holds more images than one solve should use: the sharp frames
of a walk, the frames of one rig run, a few hand-picked photographs. This draft
lets a workspace give such a set a name, defined by image patterns and frame
ranges in a small file beside the workspace file, so that a command can be
pointed at the set by name instead of by paths and ranges repeated on every
command line.

It amends these standing specs, each of which links back here:

- [`../workspace/workspace.md`](../workspace/workspace.md): the
  `.sfm-named-image-lists.json` file beside `.sfm-workspace.json`.
- [`../cli/README.md`](../cli/README.md): the `-r/--range` option that eight
  commands share.

The command that will write named image lists, a later `sfm survey`-style
command that reads the sharpness and gradient structure tensor of a capture's
frames ([image-sharpness.md](image-sharpness.md)) and proposes lists of the
sharp ones, is the intended producer; it is not designed here.

## The lists file

A workspace's named image lists live in their own file, `.sfm-named-image-lists.json`,
in the workspace root beside `.sfm-workspace.json`. The file is optional. Its
name takes the `.sfm-` prefix and the dotfile form of `.sfm-workspace.json`, so
the two sort together and stay out of an image listing, and the rest of it is
the glossary's word for what it holds, *named image list*, which is also its
key; `.sfm-image-lists.json` would drop the *named* that separates these lists
from any other list of images.

```json
{
  "version": 1,
  "named_image_lists": {
    "walk1_sharp": [
      { "pattern": "frames/walk1_%04d.jpg", "range": "1-120,131-400" }
    ],
    "rig_sharp": [
      { "pattern": "rig/*/frame_%05d.jpg", "range": "10-900:2" }
    ],
    "hero_shots": [
      { "pattern": "photos/DSC_0042.JPG" },
      { "pattern": "photos/DSC_0057.JPG" }
    ]
  }
}
```

- **`version`** (integer, required): the file's format version, `1`. A reader
  rejects a version it does not know.
- **`named_image_lists`** (object, required): from a list's name to its list of
  entries, each an object with a `pattern` and an optional `range`.

- **Name.** One or more of the ASCII letters, the digits, `-`, `_` and `.`, with
  at least one letter. Names are compared exactly, case included. The letter is
  what tells a name from a range on the command line
  ([below](#the-clis--r--range)).
- **`pattern`** (string, required): an image pattern in the `.camrig` syntax,
  relative to the workspace root: a forward-slash path with no leading `/` and
  no `..`, literal text, the glob wildcards `*` (within one path segment) and
  `**` (whole segments), and at most one frame field, `%d` or `%0Nd`, with `%%`
  for a literal `%`. One entry with a wildcard can cover every sensor of a rig,
  as `rig/*/frame_%05d.jpg` does above.
- **`range`** (string, optional): a range expression in the grammar `-r` takes
  (comma-separated integers, `a-b` spans and `a-b:s` stepped spans), applied to
  the number the pattern's frame field captures. An entry whose pattern has no
  frame field takes no `range`; one that gives one is an error. An entry with a
  frame field and no `range` takes every frame number.

**Membership.** A list's members are the union of its entries' matches: the
files under the workspace root whose workspace-relative path matches the
pattern, with the captured frame number in the entry's range. A file that two
entries of one list match is an error naming the file and both entries, since
it would leave the list's order ambiguous. Entries match files, not
directories.

**Case.** A pattern matches case-sensitively, against the paths as the
filesystem lists them or as a reconstruction records them, the same on every
platform. `photos/dsc_0042.jpg` does not match a file listed as
`photos/DSC_0042.JPG`, on Windows and macOS as on Linux, so a list selects the
same images wherever it is resolved.

**Order.** Entry order, then within an entry by frame number, then by
workspace-relative path among files of the same frame number (the sensors of
one rig frame). An entry with no frame field orders its files by
workspace-relative path. This is the order a command receives the images in,
as it receives a directory's sorted files today.

**Validation.** A command that resolves a list checks the whole file (its
version, names, patterns and ranges) and reports every
problem it finds, then matches only the list it was asked for. A list that
matches no file is an error when used. Every error names the lists file's path.
A file that is not valid JSON is reported with the line and column where
parsing failed, and nothing else in it is checked. A `version` other than `1`
is reported as "<path>: named image lists version <n> is not supported; this
sfmtool reads version 1", and nothing else in the file is checked, since its
layout is unknown. A missing `version` or `named_image_lists`, a value of the
wrong JSON type, a top-level key other than those two, and an entry key other
than `pattern` and `range` are each an error naming the key and, inside
`named_image_lists`, the list and the entry's position in it.

**Finding it.** A command first finds the workspace by the existing workspace
discovery, then reads the image lists file that sits beside that workspace's
`.sfm-workspace.json` in the workspace root. The lists file is never searched
for on its own. A command that has no workspace has no lists, so a list name
given to it is an error saying that no workspace was found and so no named
image lists are defined; a lists file with no `.sfm-workspace.json` beside it
is never used. If the workspace has no
lists file, it has no named image lists, so giving a list name is an error that
says so.

**Outside the workspace record and every `.sfmr`.** The lists file adds
nothing to `.sfm-workspace.json`, whose version 2 is about the grey conversion
alone ([image-sharpness.md](image-sharpness.md)). The lists file is not part of
`workspace.contents`, and a `.sfmr` never embeds it, records a list's name or
entries, or refers to it in any other way; a reconstruction made from a list is
the same file it would be if its images had been given by path. `sfm ws init`,
with or without `--force`, writes only `.sfm-workspace.json` and leaves the
lists file untouched.

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

- **The named lists** resolve through the `.camrig` pattern matcher in
  `sfmtool-camrig-format` (`pattern_matches`, `pattern_frame_index`) and the
  range type the CLI already parses with (`RangeExpr`, from `openjd-expr`).
  The `-r` handling moves into one shared option in
  `src/sfmtool/_commands/_range_options.py` that all eight commands use.

## Words for the glossary

To add to [`../GLOSSARY.md`](../GLOSSARY.md) when this lands:

- **named image list**: a named set of workspace images defined in
  `.sfm-named-image-lists.json` by pattern and range entries. Chosen over `image set`,
  which already names the image half of a reconstruction's data, and over
  `selection`, which the viewer uses for what is picked on screen.

## Choices made here

Decisions inside the maintainer's decisions that this draft makes and that
review should check:

1. **The lists file is named `.sfm-named-image-lists.json`**, after the glossary
   word and the form of `.sfm-workspace.json`, with its own `version` of `1`.
   A key the file does not define, at the top level or in an entry, is an
   error rather than ignored, as `.sfm-workspace.json` has no other keys.
2. **With a list, the image-directory commands take no image paths** and find
   the workspace from the current directory; narrowing a list that has an entry
   with no frame field is an error.

## Not in this draft

The command that writes named image lists, and any rule for which frames go in
one: it will be designed separately.
