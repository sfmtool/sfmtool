# Slices of a named image list (amendment)

**Status:** Draft

Amends [`../workspace/named-image-lists.md`](../workspace/named-image-lists.md),
which specifies the lists file and the resolver, and the shared `-r/--range`
option in
[`../cli/README.md`](../cli/README.md#the-shared--r--range-option); both say in
one sentence that a slice by position is proposed here, and point back here.

Decided with the maintainer: the syntax `NAME[start:stop:step]` in the value of
`-r` and `--named-image-list`; that it is exactly a Python list slice over the
list's images in the list's order, negative indexes, a negative step and
clamped bounds included; that one item is one image file, not one frame; that a
slice selects by position and never by frame number; and that it applies to
every command of the shared `-r` option. The smaller choices this draft makes
on its own are listed under [Open questions](#open-questions).

## Purpose

A named image list gives a set of a workspace's images a name, such as the
frames of a walk chosen for a solve. A user often wants part of such a list
rather than all of it: the first 30 images for a quick test solve, the last 30,
or every third image for a coarser solve. Today the only way to take part of a
list is to narrow it by a range of frame numbers (`-r walk1 -r 10-200`), which
needs the user to know which frame numbers the list holds, and a list chosen
by an algorithm holds irregular ones. A **slice** takes part of a list by
position instead, with the syntax Python uses for a list:

```bash
sfm solve -i -r 'walk1[:30]'      # the first 30 images of the list
sfm solve -i -r 'walk1[-30:]'     # the last 30
sfm solve -i -r 'walk1[::3]'      # every third image
```

## What a slice selects

**The array.** A list resolves to an ordered array of image files: its members
in the list's order, as
[named-image-lists.md § Members and their order](../workspace/named-image-lists.md#members-and-their-order)
defines it (entry order, then frame number, then path). The array is indexed
from 0. More exactly, the array a slice indexes is the images the command would
use from the list without a slice:

- For `sfm sift`, `sfm match`, `sfm motion` and `sfm solve`, the members that
  are image files, after the extension test those commands apply
  ([the shared `-r/--range` option](../cli/README.md#the-shared--r--range-option)),
  so that `walk1[:30]` is 30 images whenever the list has 30, even when its
  patterns also match files that are not images.
- For `sfm analyze --metrics`, `sfm panorama`, `sfm to-colmap-bin` and
  `sfm to-nerfstudio`, the members among the reconstruction's image names, in
  the list's order, which is what `resolve_in_names` returns. The images the
  slice keeps are then used in the reconstruction's order, as a whole list's
  are today. So a slice of a list counts only the images the reconstruction
  holds: `walk1[:30]` against a reconstruction that lacks some of the list's
  images is not the same 30 images as `walk1[:30]` given to `sfm solve`.

**Python's rules.** `NAME[start:stop:step]` keeps the items whose indexes are
`range(*slice(start, stop, step).indices(len))` in Python, for an array of
`len` items. In words:

- `step` defaults to 1. A step of 0 is an error.
- With a positive step, `start` defaults to 0 and `stop` to `len`; the items
  kept are `start`, `start + step`, … while below `stop`.
- With a negative step, `start` defaults to the last item and `stop` to before
  the first; the items kept are `start`, `start + step`, … while above `stop`,
  so the order is reversed: `walk1[::-1]` is the list backward.
- A negative `start` or `stop` counts from the end: `len` is added to it, so
  `-30` is the 30th item from the end.
- A bound beyond either end is clamped to the end, as in Python, and is not an
  error: `walk1[:30]` of a list of 20 images is all 20, and `walk1[-30:]` is
  all 20 as well.
- `NAME[:]` and `NAME[::]` are the whole list.

**Order.** The images a slice keeps are used in the slice's order: the list's
order, or the reverse for a negative step. The image-directory commands take
them in that order, as they take a whole list's members in the list's order.

**One item is one file.** An item of the array is one image file, not one frame.
In a rig list, the sensors of one frame are adjacent items when one entry
covers all the sensors (a pattern such as `rig/*/frame_%05d.jpg` orders the
files of one frame number by path). So, for a list of two sensors `cam0` and
`cam1` whose items run `cam0` 10, `cam1` 10, `cam0` 12, `cam1` 12, …:

- `[:30]` is 15 frames, both sensors of each;
- `[:31]` is those 15 frames and the `cam0` image of the 16th;
- `[::2]` is the `cam0` image of every frame and `[1::2]` the `cam1` image,
  and any even step takes one sensor only.

A slice whose `start` and `stop` are multiples of the sensor count, with a step
of 1, keeps whole frames. A step other than 1 or -1 never keeps two adjacent
items, so it keeps no frame whole: to thin a rig list by frames, narrow it by a
stepped range instead (`-r rig_sharp -r 10-900:6`). A list with one entry per
sensor holds all of the first sensor's images before any of the second's, so
`[:30]` of such a list is 30 images of the first sensor.

**By position, not by frame number.** A slice never looks at frame numbers.
That is the difference from narrowing by a range: `-r walk1 -r 10-20` keeps the
members whose frame number is between 10 and 20, whatever their positions,
while `-r 'walk1[10:20]'` keeps the 11th to the 20th images of the list,
whatever their frame numbers. A slice works on a list whose entries have no
frame field, which narrowing refuses.

**With narrowing.** A name with a slice and a range may be given together
(`-r 'walk1[:30]' -r 100-900`). The range narrows the list first, as it does
today, and the slice then indexes the narrowed array: the first 30 of the
members whose frame number is in 100-900. The other order, slicing first and
then narrowing, would make the result depend on frame numbers the slice
discarded, and is not offered. [Open questions](#open-questions) lists the
alternative of making the combination an error.

## Syntax

The value of `-r` or `--named-image-list` is a list name, optionally followed
by one slice:

```text
NAME
NAME[start:stop]
NAME[start:stop:step]
```

- **Telling a slice from a name.** A list name holds only the ASCII letters,
  the digits, `-`, `_` and `.`
  ([`is_named_image_list_name`](../../crates/sfmtool-core/src/workspace/named_image_lists.rs)),
  and a range expression only digits, `-`, `:`, `,` and whitespace, so neither
  can contain `[`. A value with a `[` in it is a name followed by a slice: the
  text before the first `[` is the name, checked by the name rule as a bare
  name is, and the text from `[` to the end is the slice.
- **The parts.** `start`, `stop` and `step` are each empty or an integer: ASCII
  digits with an optional leading `+` or `-`. A part that does not fit in a
  64-bit signed integer is an error; no list has that many images, so a bound
  that large would only ever be clamped.
- **One or two colons.** `[start:stop]` and `[start:stop:step]`, with each part
  optional. A single index with no colon, `NAME[5]`, is not accepted: in Python
  `a[5]` is one item rather than a list, and an index past the end is an
  `IndexError` rather than clamped, so accepting it would need its own rule for
  a bound out of range. `NAME[5:6]` is the sixth image. This is a choice of
  this draft ([Open questions](#open-questions)).
- **No whitespace.** A slice is written without spaces: a space anywhere in
  the brackets is an error saying so. A value with a space must be quoted to
  reach the command as one argument, and an error is clearer than a rule about
  where spaces may go.
- **One slice.** The `]` ends the value. Python allows `a[:30][::2]`; this
  draft does not, so text after the `]`, a second slice included, is an error.

## Errors

Each error names the value as given. Those about the slice's text are usage
errors, raised before any file is read:

- **Malformed.** A part that is not an integer (`walk1[a:3]`, `walk1[1.5:]`),
  more than two colons (`walk1[1:2:3:4]`), a single index (`walk1[5]`), empty
  brackets (`walk1[]`), a missing `]` (`walk1[:30`), whitespace in the
  brackets, text after the `]`, or a part out of the 64-bit range. The message
  says what is wrong and shows the form `NAME[start:stop:step]`.
- **A step of 0.** "slice step cannot be zero", Python's own message, with the
  value.
- **A slice that is not on a name.** A value with `[` whose text before it is
  not a valid name: a range (`-r '10-20[::2]'`), nothing (`-r '[:30]'`), or a
  name that breaks the name rule. The message says that a slice follows the
  name of a named image list, and for a range that a range is narrowed by
  giving it beside a name (`-r NAME -r RANGE`).

One error comes from resolving the list:

- **A slice that leaves no images.** A slice that keeps no item, such as
  `walk1[40:]` of a list of 30 images or `walk1[5:2]`, is an error naming the
  lists file, the list, the slice and the number of images the array held, in
  the manner of the empty-list errors of
  [named-image-lists.md § Members and their order](../workspace/named-image-lists.md#members-and-their-order):
  "`<path>`: the slice `[40:]` of named image list 'walk1' leaves no images; the
  list has 30". With a narrowing range the message names the range too. A list
  that is already empty, or that a range narrows to nothing, gives the errors it
  gives today, before the slice is applied.

## Quoting in a shell

`[` and `]` are glob characters in the common shells, so the documentation
always quotes a value with a slice: `-r 'walk1[:30]'`. Unquoted:

- **zsh** stops with "no matches found" when no file matches the pattern, its
  default `NOMATCH` behaviour, and the command does not run.
- **bash** passes the value unchanged when no file matches, but when a file
  does it passes that file's name instead. In a directory holding a file
  `walk13`, `-r walk1[:30]` reaches the command as `-r walk13`, since the
  bracket expression `[:30]` matches one of the characters `:`, `3` and `0`;
  the command then reports an unknown list `walk13`, or, when the workspace
  defines a list of that name, uses it.
- **PowerShell** does not expand wildcards in the arguments of a native
  command, so `sfm` receives the value as typed, quoted or not. A program built
  on the MSYS runtime (Git for Windows' tools) expands them itself, but `sfm`
  is a Python program and does not.

Quoted, the value passes through `pixi run` unchanged in bash and PowerShell.

## Uses

- **A quick test subset.** `-r 'walk1[:30]'` and `-r 'walk1[-30:]'` solve the
  start and the end of a list before a long solve of the whole of it, and
  `-r 'walk1[241:271]'` 30 images from the middle of a list of 512. A test
  subset needs no list of its own in the lists file.
- **A coarser list from a finer one.** `-r 'walk1[::3]'` keeps every third
  image. This is not the same as choosing frames with a step three times as
  large: no two images of the sliced list were checked against each other, only
  each against its neighbour in the finer list, so neighbouring images of the
  slice may not match well. Where the finer list is dense because the capture
  was hard to follow there, which is where a picking algorithm puts its short
  steps, the slice thins it as much as everywhere else, so the sliced list is
  short of images where the capture most needs them. For a rig list, a step
  separates the sensors of a frame
  ([What a slice selects](#what-a-slice-selects)).

## Finding a list's length

A slice from the middle of a list needs the list's length. No command prints a
list's members or their count on its own. The commands that resolve a list
print the count as part of their work: `sfm sift --extract -r NAME` prints
"Given named image list 'NAME', expanded to N filename(s)" before it extracts,
and `sfm analyze --metrics`, `sfm panorama`, `sfm to-colmap-bin` and
`sfm to-nerfstudio` print how many of a reconstruction's images the list
kept. The proposed
[`sfm pick-images`](pick-images-command.md) prints the length of the list it
writes, with slices to try. From Python,
`len(NamedImageLists.read(workspace_dir).resolve("walk1"))` is the number of
members, before the extension test. A command that prints a list's members and
their count is listed under [Open questions](#open-questions).

## Where the code lives

The slice is parsed and applied in Rust, beside the resolver in
[`named_image_lists.rs`](../../crates/sfmtool-core/src/workspace/named_image_lists.rs),
and bound in `sfmtool.workspace`, following
[STANDARDS.md § Logic lives in Rust](../STANDARDS.md#logic-lives-in-rust-not-in-python-wrappers):

```rust
/// A slice of a named image list's images by position, Python's
/// `[start:stop:step]`. `step` is never `Some(0)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NamedImageListSlice {
    pub start: Option<i64>,
    pub stop: Option<i64>,
    pub step: Option<i64>,
}

impl NamedImageListSlice {
    /// The bracketed text, `[` and `]` included.
    pub fn parse(text: &str) -> Result<Self, NamedImageListSliceError>;
    /// The indexes it keeps in an array of `len` items, in its order,
    /// as Python's `range(*slice(start, stop, step).indices(len))`.
    pub fn indices(&self, len: usize) -> Vec<usize>;
}

impl fmt::Display for NamedImageListSlice { /* as `[start:stop:step]` */ }

/// A `-r` or `--named-image-list` value: the list's name and its slice.
pub fn split_named_image_list_value(
    value: &str,
) -> Result<(&str, Option<NamedImageListSlice>), NamedImageListSliceError>;

impl NamedImageLists {
    /// The items `slice` keeps of `items`, the images a command takes from
    /// the list `name` (already resolved and narrowed); `SlicedToNothing`
    /// when it keeps none.
    pub fn slice_members<T>(
        &self,
        name: &str,
        narrow: Option<&RangeExpr>,
        slice: &NamedImageListSlice,
        items: Vec<T>,
    ) -> Result<Vec<T>, NamedImageListsError>;
}
```

`NamedImageListSliceError` names the value and what is wrong with it, one
variant per error under [Errors](#errors) about the slice's text.
`NamedImageListsError` gains `SlicedToNothing`, with the list, the narrowing
range, the slice and the array's length, so its message names the lists file
like every other variant's.

**Why the slice is not an argument of the resolvers.** The image-directory
commands apply their extension test to the resolver's members, in
[`_range_options.py`](../../src/sfmtool/_commands/_range_options.py)
(`resolve_list_images`), and the slice must index what that test leaves. So
the slice is a separate step after resolution, generic over the items so that
it takes paths as well as members, while the arithmetic and the messages stay
in Rust. Moving the extension test into the resolver, so that the slice could
be an argument beside `narrow`, is listed under
[Open questions](#open-questions).

**The Python side.** `parse_image_selection` in `_range_options.py` reads a
value with `[` as a name and a slice through `split_named_image_list_value`,
and `ImageSelection` carries the slice beside `list_name`, with
`describe_list` naming it ("named image list 'walk1' narrowed by the range
'100-900' and sliced by `[:30]`"). `resolve_list_images` applies
`slice_members` to the image paths after the extension test, and
`reconstruction_list_indices` to the members `resolve_in_names` returns before
it maps them to the reconstruction's indexes. Every command of the shared
option goes through these two functions, so each takes a slice with no change
of its own, `--named-image-list 'walk1[:30]'` included. The `--filter-points`
flag of `sfm to-colmap-bin` and `sfm to-nerfstudio` then drops the 3D points
the sliced list leaves without observations, as it does for a whole list. The
`--help` text of `-r` and `--named-image-list` gains a sentence on the slice.

**Filing.** When this ships, the slice's rules fold into
[named-image-lists.md](../workspace/named-image-lists.md) (a section after
"Members and their order", and the API) and into the `-r` section of
[`../cli/README.md`](../cli/README.md#the-shared--r--range-option), and
[GLOSSARY.md](../GLOSSARY.md)'s "Named image lists" table gains a row for
**slice**.

## Open questions

- **A single index.** This draft rejects `NAME[5]` and asks for `NAME[5:6]`.
  Accepting it as one image would need a rule for an index past the end, which
  Python makes an error rather than clamping.
- **A slice with a narrowing range.** This draft narrows first and slices the
  narrowed array. The alternative is to make giving both an error, so that a
  value never mixes selection by frame number with selection by position.
- **The array of a reconstruction command.** This draft slices the list's
  members among the reconstruction's images, so the same slice can name
  different images for `sfm solve` and for `sfm to-colmap-bin`. The
  alternative is to resolve the list against the workspace's files as well and
  slice that array, which needs the image files present.
- **Where the extension test lives.** Moving it into the resolver would let the
  slice be a resolver argument beside `narrow`, and would put a rule now held
  in Python into Rust.
- **Printing a list.** A way to print a list's members and their count without
  acting on them, for choosing a slice from the middle of a list: a command
  such as `sfm ws lists`, or an option of an existing command.
