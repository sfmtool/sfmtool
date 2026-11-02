# Named Image Lists

A capture often holds more images than one solve should use: the sharp frames
of a walk, the frames of one rig run, a few hand-picked photographs. A
workspace can give such a set a name in a small JSON file in its root,
`.sfm-named-image-lists.json`, which defines each set by image patterns and
frame ranges. This spec describes that file, the rules it is checked against,
and how one list resolves to the workspace's image files or to the image names
a reconstruction records. The set is called a **named image list**
([GLOSSARY.md](../GLOSSARY.md)).

The command-line side, `-r/--range` accepting a list's name on the eight
commands that take a range, is proposed in
[`../drafts/named-image-lists.md`](../drafts/named-image-lists.md).

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
has such an entry is an error naming the entry.

**An empty list.** A list that matches no file, after any narrowing, is an
error when it is resolved. A list with no entries is valid in the file and
is that error when used.

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

## Rust API

The reader, validator and resolver live in
[named_image_lists.rs](../../crates/sfmtool-core/src/workspace/named_image_lists.rs),
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
}

pub struct NamedImageListMember {
    pub path: String,       // workspace-relative, forward slashes
    pub entry: usize,       // the entry that matched it, from 0
    pub frame: Option<i64>, // None for an entry without a frame field
}
```

`NamedImageList` gives a list's `name()` and `entries()`, and
`NamedImageListEntry` an entry's `pattern()`, `range()` and
`has_frame_field()`. `RangeExpr` is `openjd_expr::RangeExpr`, the type the
`sfmtool.reconstruction.RangeExpr` binding wraps, so a list's ranges and the
CLI's `-r` values are one grammar.

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

`NamedImageListsError` is an enum whose every variant's message names the
lists file: `NotAWorkspace`, `Io`, `InvalidJson` (with `line` and `column`),
`UnsupportedVersion`, `Invalid` (with every problem), `UnknownList` (with the
names the file defines), `NarrowedEntryWithoutFrameField`, `DuplicateMatches`
(with every file two entries match), `FrameNumberTooLarge` (a frame field
that captured a number beyond `i64`), `NoMatch` and `Cancelled`.

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
`resolve_in_names(name, image_names, narrow=None)`. Both resolvers return a
list of `NamedImageListMember`, with `path`, `entry` and `frame`. `narrow` is a
`sfmtool.reconstruction.RangeExpr`, and `progress` a `ProgressCounter` that
counts the directory entries read. The module also holds
`is_named_image_list_name(name)` and the constant
`NAMED_IMAGE_LISTS_FILE_NAME`. An I/O failure raises the `OSError` subclass
for its kind, a cancelled resolve `InterruptedError`, and every other error
`ValueError`, with the Rust message.

```python
from sfmtool.reconstruction import RangeExpr
from sfmtool.workspace import NamedImageLists

lists = NamedImageLists.read(workspace_dir)
members = lists.resolve("rig_sharp", narrow=RangeExpr("10-20"))
paths = [m.path for m in members]
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
  another case. Each directory's listing is read once per resolve, so a list
  of many single-file entries in one folder reads that folder once. A `**`
  does not descend through a symbolic link to a directory, so a link back up
  the tree cannot make the walk endless; a named or `*` segment does follow
  one. Names that are not UTF-8 cannot match a pattern and are skipped. Every
  candidate is confirmed against the whole pattern before it is a member.
- **Duplicates are found before narrowing.** Two entries that match one file
  are an error whether or not a narrowing range would keep that file, since
  the list itself is ambiguous.
