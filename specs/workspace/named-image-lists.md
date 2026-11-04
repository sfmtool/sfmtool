# Named Image Lists

A capture often holds more images than one solve should use: the sharp frames
of a walk, the frames of one rig run, a few hand-picked photographs. A
workspace can give such a set a name in a small JSON file in its root,
`.sfm-named-image-lists.json`, which defines each set by image patterns and
frame ranges. This spec describes that file, the rules it is checked against,
how one list resolves to the workspace's image files or to the image names
a reconstruction records, the slice that takes part of a list by position, and
how lists are added to the file or replaced in it.
The set is called a **named image list** ([GLOSSARY.md](../GLOSSARY.md)).

The eight commands that take `-r/--range` accept a list's name there, with
`--named-image-list` as the long form, and a [slice](#slices) after the name
(`-r 'walk1[:30]'`); how they reach a list and which of its members they use
is in [the commands](#the-commands) below and in
[`../cli/README.md`](../cli/README.md#the-shared--r--range-option).

## The file

A workspace's named image lists live in `.sfm-named-image-lists.json`, in the
workspace root beside `.sfm-workspace.json` ([workspace.md](workspace.md)). The
file is optional. Its name takes the `.sfm-` prefix and the dotfile form of
`.sfm-workspace.json`, so the two sort together and stay out of an image
listing.

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

- **`version`** (integer, required): the file's format version, `1`.
- **`named_image_lists`** (object, required): from a list's name to its list of
  entries, each an object with a `pattern` and an optional `range`. The lists
  keep the file's order.
- **Name.** One or more of the ASCII letters, the digits, `-`, `_` and `.`, with
  at least one letter. Names are compared exactly, case included. The letter is
  what tells a name from a range expression, which holds only digits, `-`, `:`,
  `,` and whitespace.
- **`pattern`** (string, required): an image pattern in the `.camrig` syntax
  ([camrig-file-format.md](../formats/camrig-file-format.md#how-camrig-files-fit-into-workspaces)),
  relative to the workspace root: a forward-slash path with no leading `/` and
  no `..`, literal text, the glob wildcards `*` (within one path segment) and
  `**` (whole segments), and at most one frame field, `%d` or `%0Nd`, with `%%`
  for a literal `%`. One entry with a wildcard can cover every sensor of a rig,
  as `rig/*/frame_%05d.jpg` does above.
- **`range`** (string, optional): a range expression (comma-separated integers,
  `a-b` spans and `a-b:s` stepped spans), applied to the number the pattern's
  frame field captures. An entry whose pattern has no frame field takes no
  `range`. An entry with a frame field and no `range` takes every frame number.

The file holds nothing else: no other top-level key and no other entry key.

## Members and their order

**Membership.** A list's members are the union of its entries' matches. An
entry matches a file when the file's workspace-relative, forward-slash path
matches the pattern and the frame number the frame field captures is in the
entry's range. A frame field is greedy, as in `printf`: it takes the longest
run of digits after which the rest of the pattern still matches. A file that
two entries of one list match is an error naming the file and both entries,
since it would leave the list's order ambiguous. Entries match files, not
directories.

**Case.** A pattern matches case-sensitively, against the paths as the
filesystem lists them or as a reconstruction records them, the same on every
platform. `photos/dsc_0042.jpg` does not match a file listed as
`photos/DSC_0042.JPG`, on Windows and macOS as on Linux, so a list selects the
same images wherever it is resolved.

**Order.** Entry order, then within an entry by frame number, then by
workspace-relative path among files of the same frame number (the sensors of
one rig frame). An entry with no frame field orders its files by
workspace-relative path. Paths compare byte by byte, so `B` sorts before `a`.

**Narrowing.** A caller may narrow a list by a further range: the members are
then the list's members whose frame number that range also holds. An entry
with no frame field has no frame number to narrow, so narrowing a list that
has such an entry is an error naming the entry. Narrowing selects members by
frame number only, never by their position in the list; a
[slice](#slices) selects by position.

**An empty list.** A list that matches no file, after any narrowing, is an
error when it is resolved, naming the list. When the list matches files and
the narrowing range holds the frame number of none of them, the error says
instead that the range left no members, naming the list and the range. A list
with no entries is valid in the file and is the first error when used.

## Slices

A user often wants part of a list rather than all of it: the first 30 images
for a quick test solve, the last 30, or every third image for a coarser solve.
A **slice** takes part of a list by position, with the syntax and the rules of
a Python list slice, written after the list's name:

```bash
sfm solve -i -r 'walk1[:30]'      # the first 30 images of the list
sfm solve -i -r 'walk1[-30:]'     # the last 30
sfm solve -i -r 'walk1[::3]'      # every third image
```

### What a slice selects

**The array.** A slice indexes, from 0, the ordered array of image files a
command would take from the list without a slice: the list's members in the
list's order ([Members and their order](#members-and-their-order)), narrowed
first by a range when one is given, and reduced to what the command uses:

- For `sfm sift`, `sfm match`, `sfm motion` and `sfm solve`, the members that
  are image files, after the extension test those commands apply
  ([the commands](#the-commands)), so `walk1[:30]` is 30 images whenever the
  list has 30, even when its patterns also match files that are not images.
- For `sfm analyze --metrics`, `sfm panorama`, `sfm to-colmap-bin` and
  `sfm to-nerfstudio`, the members among the reconstruction's image names, in
  the list's order, as `resolve_in_names` returns them. The images the slice
  keeps are then used in the reconstruction's order, as a whole list's are.
  So a slice counts only the images the reconstruction holds: `walk1[:30]`
  against a reconstruction that lacks some of the list's images is not the
  same 30 images as `walk1[:30]` given to `sfm solve`.

**Python's rules.** `NAME[start:stop:step]` keeps the items whose indexes are
`range(*slice(start, stop, step).indices(len))` in Python, for an array of
`len` items:

- `step` defaults to 1. A step of 0 is an error.
- With a positive step, `start` defaults to 0 and `stop` to `len`; the items
  kept are `start`, `start + step`, … while below `stop`.
- With a negative step, `start` defaults to the last item and `stop` to before
  the first; the items kept are `start`, `start + step`, … while above `stop`,
  so the order is reversed: `walk1[::-1]` is the list backward.
- A negative `start` or `stop` counts from the end: `len` is added to it, so
  `-30` is the 30th item from the end.
- A bound beyond either end is clamped to that end and is not an error:
  `walk1[:30]` of a list of 20 images is all 20, and so is `walk1[-30:]`.
- `NAME[:]` and `NAME[::]` are the whole list.

**Order.** The images a slice keeps are used in the slice's order: the list's
order, or its reverse for a negative step. The image-directory commands take
them in that order, as they take a whole list's members in the list's order.

**One item is one file.** An item of the array is one image file, not one
frame. In a rig list, the sensors of one frame are adjacent items when one
entry covers all the sensors, since a pattern such as `rig/*/frame_%05d.jpg`
orders the files of one frame number by path. So, for a list of two sensors
`cam0` and `cam1` whose items run `cam0` 10, `cam1` 10, `cam0` 12, `cam1` 12,
…:

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
`-r walk1 -r 10-20` keeps the members whose frame number is between 10 and
20, whatever their positions, while `-r 'walk1[10:20]'` keeps the 11th to the
20th images of the list, whatever their frame numbers. A slice works on a list
whose entries have no frame field, which narrowing refuses.

**With narrowing.** A name with a slice and a range may be given together
(`-r 'walk1[:30]' -r 100-900`). The range narrows the list first, and the
slice then indexes the narrowed array: the first 30 of the members whose frame
number is in 100-900. Slicing first and narrowing after is not offered, since
the result would then depend on frame numbers the slice discarded.

**A coarser list from a finer one.** `-r 'walk1[::3]'` keeps every third
image. That is not the same as choosing frames with a step three times as
large: no two images of the sliced list were checked against each other, only
each against its neighbour in the finer list, so neighbouring images of the
slice may not match well. Where the finer list is dense because the capture
was hard to follow there, the slice thins it as much as everywhere else.

**A list's length.** A slice from the middle of a list needs the list's
length. The commands that resolve a list print the count as part of their
work: `sfm sift --extract -r NAME` prints "Given named image list 'NAME',
expanded to N filename(s)" before it extracts, and `sfm analyze --metrics`,
`sfm panorama`, `sfm to-colmap-bin` and `sfm to-nerfstudio` print how many of
a reconstruction's images the list kept. From Python,
`len(NamedImageLists.read(workspace_dir).resolve("walk1"))` is the number of
members, before the extension test.

### Syntax

The value of `-r` or `--named-image-list` is a list name, optionally followed
by one slice:

```text
NAME
NAME[start:stop]
NAME[start:stop:step]
```

- **Telling a slice from a name.** A list name holds only the ASCII letters,
  the digits, `-`, `_` and `.`, and a range expression only digits, `-`, `:`,
  `,` and whitespace, so neither can contain `[`. A value with a `[` in it is
  a name followed by a slice: the text before the first `[` is the name,
  checked by the name rule as a bare name is, and the text from `[` to the end
  is the slice.
- **The parts.** `start`, `stop` and `step` are each empty or an integer:
  ASCII digits with an optional leading `+` or `-`. A part that does not fit
  in a 64-bit signed integer is an error; no list has that many images, so a
  bound that large would only ever be clamped.
- **One or two colons.** `[start:stop]` and `[start:stop:step]`, with each part
  optional. A single index with no colon, `NAME[5]`, is not accepted: in
  Python `a[5]` is one item rather than a list, and an index past the end is
  an `IndexError` rather than clamped. `NAME[5:6]` is the sixth image, and the
  error for `NAME[5]` says so.
- **No whitespace.** A slice is written without spaces: a space anywhere in
  the brackets is an error saying so.
- **One slice.** The `]` ends the value. Python allows `a[:30][::2]`; a value
  here does not, so text after the `]`, a second slice included, is an error.

### Errors

Each error names the value as given. Those about the slice's text are usage
errors, raised before any file is read:

- **Malformed.** A part that is not an integer (`walk1[a:3]`, `walk1[1.5:]`),
  more than two colons (`walk1[1:2:3:4]`), a single index (`walk1[5]`), empty
  brackets (`walk1[]`), a missing `]` (`walk1[:30`), whitespace in the
  brackets, text after the `]`, or a part out of the 64-bit range. The message
  says what is wrong and shows the form `NAME[start:stop:step]`.
- **A step of 0.** "slice step cannot be zero", Python's own message, after
  the value.
- **A slice that is not on a name.** A value with `[` whose text before it is
  not a valid name: a range (`-r '10-20[::2]'`), nothing (`-r '[:30]'`), or a
  name that breaks the name rule. The message says that a slice follows the
  name of a named image list, and for a range that a list is narrowed by a
  range given beside its name (`-r NAME -r RANGE`).

One error comes from applying the slice:

- **A slice that leaves no images.** A slice that keeps no item, such as
  `walk1[40:]` of a list of 30 images or `walk1[5:2]`, is an error naming the
  lists file, the list, the slice and the number of images the array held:
  "`<path>`: the slice [40:] of named image list 'walk1' leaves none of its 30
  images". With a narrowing range the message names the range too ("… of
  named image list 'walk1' narrowed by the range '100-900' leaves none of
  …"). A list that is already empty, or that a range narrows to nothing, gives
  the errors of [an empty list](#members-and-their-order), before the slice is
  applied.

### Quoting in a shell

`[` and `]` are glob characters in the common shells, so a value with a slice
is quoted: `-r 'walk1[:30]'`. Unquoted:

- **zsh** stops with "no matches found" when no file matches the pattern, its
  default `NOMATCH` behaviour, and the command does not run.
- **bash** passes the value unchanged when no file matches, but when a file
  does it passes that file's name instead. In a directory holding a file
  `walk13`, `-r walk1[:30]` reaches the command as `-r walk13`, since the
  bracket expression `[:30]` matches one of the characters `:`, `3` and `0`;
  the command then reports an unknown list `walk13`, or, when the workspace
  defines a list of that name, uses it.
- **PowerShell** does not expand wildcards in the arguments of a native
  command, so `sfm` receives the value as typed, quoted or not. A program
  built on the MSYS runtime (Git for Windows' tools) expands them itself, but
  `sfm` is a Python program and does not.

Quoted, the value passes through `pixi run` unchanged in bash and PowerShell.

## Validation

Reading the file checks the whole of it, and reports every problem it finds,
each on its own line, before any list is resolved. Every message starts with
the file's path. An entry is named by its list and its position in the list,
counted from 1:

```text
ws/.sfm-named-image-lists.json: list 'walk1_sharp' entry 2: 'range' is a number, not a string
```

- **Not JSON.** A file that is not valid JSON is reported with the line and
  column where parsing failed, and nothing else in it is checked. A leading
  UTF-8 byte order mark is skipped.
- **Another version.** A `version` that is an integer other than `1` is
  reported as "`<path>`: named image lists version `<n>` is not supported; this
  sfmtool reads version 1", and nothing else in the file is checked, since its
  layout is unknown.
- **Keys.** A missing `version`, `named_image_lists` or `pattern`, a value of
  the wrong JSON type, a top-level key other than those two, an entry key other
  than `pattern` and `range`, and a key or list name that appears twice in one
  object are each an error naming the key.
- **Names, patterns and ranges.** A name that breaks the name rule, a pattern
  the `.camrig` pattern checker rejects (its reason is the message), a range
  that does not parse, and a `range` on an entry whose pattern has no frame
  field are each an error.

## Writing the file

Lists are added to the file, or replaced in it, by
[`NamedImageLists::write_lists`](#rust-api), which creates the file when the
workspace has none. The new lists are checked first, by the rules of
[Validation](#validation), and a name given twice among them is an error too;
then the existing file is read and checked whole, as a command reads it. A
name the file already defines is refused, naming every such list, unless the
caller asks to replace it. Each refusal names the file, and nothing is written
when any check fails, so a file that breaks the rules is left for a person to
fix rather than replaced.

**The file is rewritten whole.** Every list it held keeps its place and order,
a replaced list takes the place of the one it replaces, and new lists follow in
the order given. The text is written in one layout, the one of the example in
[The file](#the-file): two-space indents and one line per entry, `pattern`
before `range`, ending in a newline. A file in that layout, as the writer
leaves it, is rewritten byte for byte except for the lists that change. A file
laid out by hand loses its whitespace and its key order within an entry, and a
byte order mark, but nothing it means: the format allows no other key, so the
version, the lists and each entry's pattern and range are the whole of its
content. Each existing range is written from its text in the file, never from
its parsed form, so `1,2,3` stays `1,2,3` rather than becoming `1-3`. A new
range is written as `RangeExpr`'s `Display` writes it, and that text must
parse back as a range, so a range of more sub-ranges than a range expression
may hold is refused rather than written.

**A failed write leaves the old file.** The text is written to a temporary
file beside the lists file and renamed over it, which replaces an existing file
on every platform. A write that fails at any point removes the temporary file
and leaves the old lists file as it was. On Windows, a lists file that another
process holds open without sharing it for deletion cannot be replaced, and the
write fails with an I/O error naming the file.

The check for a defined name and the write are not one step: a list another
process adds between them is lost when the file is written.

## Finding the file

A command finds the workspace by the upward search for `.sfm-workspace.json`
([workspace.md](workspace.md#workspace-discovery)), then reads the lists file
that sits beside it in the workspace root. The lists file is never searched
for on its own, and one with no `.sfm-workspace.json` beside it is never used.
A workspace with no lists file has no named image lists, which is a different
state from a file that defines none.

The lists file is not part of `.sfm-workspace.json` or of the snapshot of it a
`.sfmr` embeds (`workspace.contents`). A `.sfmr` never embeds it, records a
list's name or entries, or refers to it in any other way, and `sfm ws init`
leaves it untouched.

## The commands

`sfm sift`, `sfm match`, `sfm motion` and `sfm solve` resolve a list with
`resolve`, in the workspace the upward search from the current directory
finds, and use the list's members in place of image paths. They use only the
members that are image files, by the same extension test their directory
expansion applies, and skip the rest. `sfm analyze --metrics`, `sfm panorama`,
`sfm to-colmap-bin` and `sfm to-nerfstudio` resolve a list with
`resolve_in_names` against the reconstruction's image names, in the
reconstruction's workspace. A name given with a range is resolved with the
range as `narrow`. A value with a slice is split into the name and the slice
by `split_named_image_list_value` before anything is read, and the slice is
applied with `slice_members` to the images the command takes from the
resolved list: the image paths after the extension test, or the members
`resolve_in_names` returns, before they are put in the reconstruction's order.
The rules for telling a name from a range and the errors are in
[`../cli/README.md`](../cli/README.md#the-shared--r--range-option).

## Rust API

The reader, validator and resolver live in
[named_image_lists.rs](../../crates/sfmtool-core/src/workspace/named_image_lists.rs),
the slice's parser and arithmetic in
[named_image_lists/slice.rs](../../crates/sfmtool-core/src/workspace/named_image_lists/slice.rs),
and the writer in
[named_image_lists/write.rs](../../crates/sfmtool-core/src/workspace/named_image_lists/write.rs),
in `sfmtool_core::workspace`, bound as `sfmtool.workspace`.

```rust
pub const NAMED_IMAGE_LISTS_FILE_NAME: &str = ".sfm-named-image-lists.json";
pub const WORKSPACE_FILE_NAME: &str = ".sfm-workspace.json";
pub const NAMED_IMAGE_LISTS_VERSION: i64 = 1;

pub fn is_named_image_list_name(name: &str) -> bool;

impl NamedImageLists {
    /// `Ok(None)` when the workspace has no lists file.
    pub fn read(workspace_dir: &Path) -> Result<Option<Self>, NamedImageListsError>;
    pub fn parse(workspace_dir: &Path, bytes: &[u8]) -> Result<Self, NamedImageListsError>;
    pub fn path(&self) -> &Path;
    pub fn workspace_dir(&self) -> &Path;
    pub fn lists(&self) -> &[NamedImageList];
    pub fn get(&self, name: &str) -> Option<&NamedImageList>;

    /// Against the files under the workspace root.
    pub fn resolve(
        &self,
        name: &str,
        narrow: Option<&RangeExpr>,
        progress: &Progress,
    ) -> Result<Vec<NamedImageListMember>, NamedImageListsError>;

    /// Against the workspace-relative image names a reconstruction records.
    pub fn resolve_in_names<S: AsRef<str>>(
        &self,
        name: &str,
        image_names: &[S],
        narrow: Option<&RangeExpr>,
    ) -> Result<Vec<NamedImageListMember>, NamedImageListsError>;

    /// The items `slice` keeps of `items`, the images a command takes from
    /// the list `name` (already resolved and narrowed by `narrow`), in the
    /// slice's order; `SlicedToNothing` when it keeps none.
    pub fn slice_members<T>(
        &self,
        name: &str,
        narrow: Option<&RangeExpr>,
        slice: &NamedImageListSlice,
        items: Vec<T>,
    ) -> Result<Vec<T>, NamedImageListsError>;

    /// Add `lists` to the workspace's lists file, or replace lists of the
    /// same names when `replace`, creating the file when absent; the file's
    /// path.
    pub fn write_lists(
        workspace_dir: &Path,
        lists: &[NamedImageList],
        replace: bool,
    ) -> Result<PathBuf, NamedImageListsError>;
}

impl NamedImageList {
    /// Checked when written, not here.
    pub fn new(name: impl Into<String>, entries: Vec<NamedImageListEntry>) -> Self;
}

impl NamedImageListEntry {
    /// Checked when written, not here; `range` is written by its `Display`.
    pub fn new(pattern: impl Into<String>, range: Option<RangeExpr>) -> Self;
}

pub struct NamedImageListMember {
    pub path: String,       // workspace-relative, forward slashes
    pub entry: usize,       // the entry that matched it, from 0
    pub frame: Option<i64>, // None for an entry without a frame field
}

/// Python's `[start:stop:step]`; each part `None` when left empty, and the
/// step never `Some(0)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NamedImageListSlice { /* start, stop, step: Option<i64> */ }

impl NamedImageListSlice {
    /// `ZeroStep` for a step of `Some(0)`.
    pub fn new(start: Option<i64>, stop: Option<i64>, step: Option<i64>)
        -> Result<Self, NamedImageListSliceError>;
    /// The bracketed text, `[` and `]` included.
    pub fn parse(text: &str) -> Result<Self, NamedImageListSliceError>;
    pub fn start(&self) -> Option<i64>;
    pub fn stop(&self) -> Option<i64>;
    pub fn step(&self) -> Option<i64>;
    /// Python's `range(*slice(start, stop, step).indices(len))`.
    pub fn indices(&self, len: usize) -> Vec<usize>;
}

impl fmt::Display for NamedImageListSlice { /* `[start:stop]` or `[start:stop:step]` */ }

/// A `-r` or `--named-image-list` value: the list's name and its slice.
pub fn split_named_image_list_value(
    value: &str,
) -> Result<(&str, Option<NamedImageListSlice>), NamedImageListSliceError>;
```

`NamedImageList` gives a list's `name()` and `entries()`, and
`NamedImageListEntry` an entry's `pattern()`, `range()` and
`has_frame_field()`. An entry read from the file also keeps its range's text
as the file wrote it, which is what `write_lists` writes back. `RangeExpr` is
`openjd_expr::RangeExpr`, the type the `sfmtool.reconstruction.RangeExpr`
binding wraps, so a list's ranges and the CLI's `-r` values are one grammar.

**Why it is shaped this way.** A `NamedImageLists` value exists only once the
whole file has passed validation, so a resolver never meets a bad entry and
every command that reads the file reports the same problems. The value holds
the workspace root it was read from, so `resolve` takes no second path that
could disagree with the file's. The workspace root is the caller's argument
rather than something `read` searches for, because the upward search already
exists, in `sfmtool._workspace.find_workspace_for_path`, and the lists file
must sit beside the workspace that search finds; `read` refuses a directory
with no `.sfm-workspace.json` in it. The two resolvers return the same
member type because a command on image directories and a command on a
reconstruction ask the same question of different candidates: the
reconstruction's names are resolved without touching the filesystem, so a list
works where the image files are absent. Narrowing is an argument of the
resolver, not a step after it, so that the frame-field check and the empty
check see the narrowed list.

`write_lists` takes the same `NamedImageList` and `NamedImageListEntry` types
the reader gives, so a list read from one file can be written to another
unchanged, range text included. Their `new` constructors do not check what
they are given, because `write_lists` checks every new list by the reader's
own entry check, on the text it will write, and reports every problem at once
in one error that names the file; a constructor that checked would report the
same problems without the file and one at a time. It is an associated function
rather than a method on a value `read` returned, because it reads and checks
the file itself just before writing, so a list another command added since a
caller's earlier `read` is kept, and a name it added is still refused.

`NamedImageListsError` is an enum whose every variant's message names the
lists file: `NotAWorkspace`, `Io`, `InvalidJson` (with `line` and `column`),
`UnsupportedVersion`, `Invalid` (with every problem), `InvalidNewLists` (with
every problem of the lists given to `write_lists`), `ListExists` (with every
list `write_lists` was given that the file already defines), `UnknownList`
(with the names the file defines), `NarrowedEntryWithoutFrameField`,
`DuplicateMatches` (with every file two entries match), `FrameNumberTooLarge` (a frame field
that captured a number beyond `i64`), `NoMatch`, `NarrowedToNothing` (with
the narrowing range), `SlicedToNothing` (with the narrowing range, the slice
and the number of items it was given) and `Cancelled`.

`NamedImageListSliceError` is the error about a value's text, and names the
value as given: `InvalidName` (a value with no `[` that breaks the name rule),
`NotOnAName` (a `[` with no valid name before it), `Malformed` (with what is
wrong) and `ZeroStep`. `split_named_image_list_value` checks a bare name as
well as a sliced one, so a caller has one function to read the value with.

**Why the slice is not an argument of the resolvers.** The image-directory
commands apply their extension test to the resolver's members, and the slice
must index what that test leaves. So the slice is a step after resolution,
generic over the items so that it takes paths as well as members, while the
arithmetic and the messages stay in Rust. `NamedImageListSlice`'s fields are
private, with `new` the only way to build one from parts, so that no caller
can make a slice with a step of 0 and `indices` never meets one.

`resolve` reports through `progress` the number of directory entries it has
read, and stops with `Cancelled` when `progress` is cancelled, since a `**`
pattern can walk a large workspace.

```rust
use sfmtool_core::workspace::NamedImageLists;
use sfmtool_core::progress::Progress;

if let Some(lists) = NamedImageLists::read(workspace_root)? {
    let members = lists.resolve("walk1_sharp", None, &Progress::none())?;
    let paths: Vec<&str> = members.iter().map(|m| m.path.as_str()).collect();
}
```

### Python

`sfmtool.workspace` binds the reader as the class `NamedImageLists`, with
`read(workspace_dir)` (a static method that returns `None` when the workspace
has no lists file), the `path`, `workspace_dir` and `names` properties,
`len()` and `in`, `resolve(name, narrow=None, progress=None)` and
`resolve_in_names(name, image_names, narrow=None)`, and
`slice_members(name, slice, items, narrow=None)`, which takes a list of any
objects. Both resolvers return a list of `NamedImageListMember`, with `path`,
`entry` and `frame`. `narrow` is a `sfmtool.reconstruction.RangeExpr`, and
`progress` a `ProgressCounter` that counts the directory entries read. The
static method `write_lists(workspace_dir, lists, replace=False)` writes the
file as [Writing the file](#writing-the-file) says and returns its path;
`lists` is a sequence of `(name, entries)` pairs, each entry a
`(pattern, range)` pair whose range is a `RangeExpr` or `None`. The
class `NamedImageListSlice` is built as `NamedImageListSlice(start=None,
stop=None, step=None)` or read by its static method `parse(text)`, has the
`start`, `stop` and `step` properties and `indices(length)`, compares and
hashes by its parts, and `str()` writes it back in brackets. The module also
holds `is_named_image_list_name(name)`,
`split_named_image_list_value(value)`, which returns the name and a
`NamedImageListSlice` or `None`, and the constant
`NAMED_IMAGE_LISTS_FILE_NAME`. An I/O failure, of a read or of a write, raises
the `OSError` subclass for its kind, a cancelled resolve `InterruptedError`,
and every other error `ValueError`, with the Rust message.

```python
from sfmtool.reconstruction import RangeExpr
from sfmtool.workspace import NamedImageLists, split_named_image_list_value

lists = NamedImageLists.read(workspace_dir)
members = lists.resolve("rig_sharp", narrow=RangeExpr("10-20"))
paths = [m.path for m in members]

name, walk_slice = split_named_image_list_value("walk1[:30]")
first_30 = lists.slice_members(name, walk_slice, lists.resolve(name))

NamedImageLists.write_lists(
    workspace_dir,
    [("walk1_picked", [("frames/walk1_%04d.jpg", RangeExpr("1-61:15,74"))])],
)
```

## Implementation notes

- **One grammar for patterns.** Patterns are checked by
  `sfmtool_camrig_format::validate_pattern` and matched by its `pattern_matches`
  and `pattern_frame_index`, always with `case_insensitive = false`, so the
  rules for a `.camrig` pattern and a list pattern cannot drift apart.
- **Keeping duplicate keys.** The file is parsed into a small JSON tree
  ([json.rs](../../crates/sfmtool-core/src/workspace/named_image_lists/json.rs))
  that keeps every member of an object in the file's order, rather than into
  `serde_json::Value`, which keeps only the last member of a repeated key. A
  list name written twice would otherwise lose its first definition without a
  word.
- **The walk follows the pattern.** The filesystem resolver
  ([walk.rs](../../crates/sfmtool-core/src/workspace/named_image_lists/walk.rs))
  lists only the directories a pattern's segments lead into, plus every
  directory below a `**`, and compares each segment with the camrig matcher
  against the names a directory lists. It never asks the filesystem whether a
  literal name exists, which on Windows and macOS would answer for a name in
  another case. Each directory's listing is read once per resolve and kept
  with a map from each name to its entry, so a list of many single-file
  entries in one folder reads that folder once, and a segment with no
  wildcard and no frame field is one exact lookup in that map rather than a
  comparison with every name the folder holds. A `**`
  does not descend through a symbolic link to a directory, so a link back up
  the tree cannot make the walk endless; a named or `*` segment does follow
  one. Names that are not UTF-8 cannot match a pattern and are skipped. Every
  candidate is confirmed against the whole pattern before it is a member.
- **Literal entries in `resolve_in_names`.** An entry whose whole pattern has
  no wildcard and no frame field (`sfmtool_camrig_format::pattern_literal`
  gives its text) is looked up in a set of the reconstruction's names built
  once per call, rather than compared with every name; other entries are
  compared with every name.
- **Duplicates are found before narrowing.** Two entries that match one file
  are an error whether or not a narrowing range would keep that file, since
  the list itself is ambiguous.
