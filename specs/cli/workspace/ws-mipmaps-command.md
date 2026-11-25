# `sfm ws mipmaps` Command

`sfm ws mipmaps` prepares, checks and cleans the mipmaps of a workspace's
large photographs. In a workspace whose `.sfm-workspace.json` has
`mipmap_options`, a photograph with a side over its `out_of_core_long_side` is
**out of core**: its reduced levels, and its full-resolution copy when the
original cannot be read a rectangle at a time, are JPEG **mip level files**
under the workspace's **mipmap directory**, `.sfm-mipmaps/`
([mipmaps.md](../../core/camera/mipmaps.md)). `sfm sift --extract`, the
viewer and every reader of the photograph cache make the mipmaps they need;
this command makes them for the photographs under its paths at once, which is
the way to prepare a workspace whose features came from somewhere else or
whose reconstruction is not open in the viewer. With `--check` it reports
which mipmaps are missing or stale and which files are orphans, writing
nothing; with `--prune` it deletes the orphan files.

`sfm ws init` does not yet write `mipmap_options`, and the command's
`--enable`, which adds the object to an existing workspace, is planned in
[out-of-core-photograph-pyramid.md](../../drafts/out-of-core-photograph-pyramid.md)
(step 17); until then a workspace has the object only when its owner wrote it.

## Command Syntax

```bash
sfm ws mipmaps PATHS... [--check | --prune]
```

| Option | Effect |
|--------|--------|
| (none) | Make the mipmap of each out-of-core photograph among the images under `PATHS` whose mip level files are missing or stale. |
| `--check` | Write nothing. Report the out-of-core photographs among the images under `PATHS` whose mipmaps are missing or stale, and the files `--prune` would delete; exit with status 1 when there is any. |
| `--prune` | Delete the orphan files of the part of each workspace's mipmap directory that mirrors `PATHS` ([Orphans](#orphans)), and make no mipmap. |

`--check` and `--prune` together are a usage error. `PATHS` is required: a
prune deletes files, so it searches only where it is told to.

## Which photographs

For a build and a check, each path that is a directory is expanded as
`sfm sift` expands one (`expand_paths` with `SIFT_IMAGE_EXTENSIONS`, `.png`,
`.jpg` and `.jpeg` without regard to case, recursively, skipping every
directory named `.sfm-mipmaps`), and a path that is a file is taken as it is,
so the two commands read the same images under the same paths. A path with
no image under it is a usage error. Each photograph belongs to the workspace
found by the upward search from it, so a path whose tree holds a nested
workspace reaches that workspace's photographs, whose mipmaps are under the
nested workspace's own mipmap directory.

Each photograph's header and its workspace's `mipmap_options` are read
first, and a photograph in core is read no further: one in a workspace
without the object, outside every workspace, or with no side over the
threshold. A photograph with a side over 65535 px, one whose file cannot be
read, and every photograph of a workspace whose `mipmap_options` is invalid
is a failure, named in a warning; it does not stop the others.

## Making the mipmaps

The build is one `PhotographCache(budget_bytes=0).build_mipmaps(images)`
([python-bindings.md § `photographs`](../../python-bindings.md#photographs)),
as `sfm sift --extract` makes its mipmaps: a mipmap already current is read
rather than made again, an out-of-core photograph whose mipmap is missing or
stale is decoded and its mip level files written, and the cache's build
budget bounds the builds in progress. The cache keeps nothing, since the
command reads nothing back.

## Checking

`--check` reads, for each out-of-core photograph, whether there is a file at
its coarsest mip level's name, and when there is, whether its mipmap is
current as [mipmaps.md § "When a mip level file is current"](../../core/camera/mipmaps.md#when-a-mip-level-file-is-current)
says: each stored level's file, from the coarsest, then `-mip0` when the
files say the original is not region-readable. A photograph is **missing**
with no coarsest file (no mipmap was made, or it was made at another
quality), **stale** when a file fails a test, the first such file and why
given, and **current** otherwise. The original is hashed only when its length
matches a record and its modification time does not. It then lists the files
a prune of the same paths would delete, from a dry run of the prune. Nothing
is written: no mip level file, no mipmap directory and no `CACHEDIR.TAG`.

## Orphans

`--prune` searches, for each path, the part of the enclosing workspace's
mipmap directory that mirrors it: under `<root>/.sfm-mipmaps/<path relative
to root>` when that is a directory, and the files beside it whose names are
the path's file name followed by `-mip` (a photograph's mip level files) or
those of a build's temporary files. A path need not exist, so the path of a
deleted photograph or directory finds the files it left. A path that is a
directory also searches the whole mipmap directory of each workspace nested
below it. A path in no workspace and with no workspace below it has nothing
to prune and is a warning.

A file found there is judged by its name first. A name that is not a mip
level file's, `<photograph>-mip<l>-q<quality>.jpg` with the level and the
quality (1 to 100) written in decimal without a leading zero, nor a build's
temporary file's, `.<mip level file name>.<process id>-<count>.tmp`, is left
as it is, as is `CACHEDIR.TAG`. A mip level file is an **orphan** when, in
this order:

1. the workspace's file has no `mipmap_options`, so the workspace keeps no
   mip level file;
2. its quality is not the one `mipmap_options` gives its level
   (`full_resolution_jpeg_quality` for `-mip0`, `jpeg_quality` otherwise);
3. its photograph, the file at the path the mipmap directory mirrors with
   the name before `-mip`, no longer exists, or is not a file;
4. its photograph is in core, or is too large to be read;
5. its photograph's mipmap is kept elsewhere, in the mipmap directory of a
   workspace nested in this one;
6. its level is over the number of levels the photograph's mipmap stores;
7. it is not current for its photograph: its frame or record fails test 1,
   its record names another original or one that has changed (tests 2 to 4
   of [when a mip level file is current](../../core/camera/mipmaps.md#when-a-mip-level-file-is-current)),
   or it records another number of stored levels;
8. it is a `-mip0` whose record says the original is region-readable, so the
   mipmap has no `-mip0`.

A temporary file is an orphan when it is more than an hour old by its
modification time (`TEMPORARY_FILE_MAX_AGE`): a build writes and renames one
within seconds, so an older one was left by a build that stopped.

Each file is judged on its own, so the current files a stopped build left are
kept, and the next build makes only the levels it did not write. Whether a
photograph is in a reconstruction plays no part: the mipmap of a photograph
that no reconstruction uses is kept while it is current. A photograph whose
file or header cannot be read keeps its files, each named in a warning, as do
a file that cannot be read and every file of a workspace whose
`mipmap_options` is invalid, whose mipmap directory is not searched.

**What is deleted.** Only a regular file inside a mipmap directory whose name
is a mip level file's or a temporary file's, judged an orphan. The search
never enters a link: a symbolic link, or on Windows a junction, met in a
mipmap directory, or a mipmap directory that is one, is listed and left as it
is, and neither it nor what it leads to is deleted. Just before each delete
the file is checked again: it must still be a regular file and not a link,
and its directory, with every link resolved, must be inside the mipmap
directory, likewise resolved. A path that leads out of its workspace through
`..` is refused with a warning. No directory is deleted, not even one the
prune empties; deleting the mipmap directory by hand is always safe.

## Output

A build prints the images found and the mipmaps made, the second number
counting the out-of-core photographs whose mipmaps are now current, made or
found so:

```
Images: 2, 1 out of core and 1 in core
New mipmaps: 1 / 1 out-of-core image(s)
```

A photograph whose mipmap could not be made is a line `Warning: no mipmap
was made: <why>` on standard error, and the command then exits with an error
counting them.

A check prints the same first line, the count of current mipmaps, and a list
of each kind of thing to do, every path relative to the current directory
when it is under it:

```
Images: 4, 3 out of core and 1 in core
Current mipmaps: 1 / 3 out-of-core image(s)
Missing mipmaps: 1
  images/IMG_0004.JPG
Stale mipmaps: 1
  images/IMG_0002.JPG: IMG_0002.JPG-mip1-q95.jpg is not current: it was made from an original of 965409 bytes, and this one has 911263
Orphan files --prune would delete: 4
  .sfm-mipmaps/images/IMG_0002.JPG-mip0-q95.jpg: it is not current: it was made from an original of 965409 bytes, and this one has 911263
  .sfm-mipmaps/images/IMG_0002.JPG-mip1-q95.jpg: it is not current: it was made from an original of 965409 bytes, and this one has 911263
  .sfm-mipmaps/images/IMG_0003.JPG-mip0-q95.jpg: its photograph images/IMG_0003.JPG no longer exists
  .sfm-mipmaps/images/IMG_0003.JPG-mip1-q95.jpg: its photograph images/IMG_0003.JPG no longer exists
```

A stale mipmap's files are orphans too, since no photograph reads them; a
prune deletes them and the next build makes the mipmap again, as it would
make it over them.

It exits with status 1 when a mipmap is missing or stale or a file is an
orphan, and 0 when every mipmap is current and nothing is an orphan, so a
script can test for a workspace that needs a build or a prune. A photograph
that could not be checked is a `Warning: not checked: <why>` line, and the
command then exits with an error.

A prune lists the files it deleted with why, and counts the mip level files
it kept:

```
Deleted 2 orphan file(s):
  .sfm-mipmaps/images/IMG_0003.JPG-mip0-q95.jpg: its photograph images/IMG_0003.JPG no longer exists
  .sfm-mipmaps/images/IMG_0003.JPG-mip1-q95.jpg: its photograph images/IMG_0003.JPG no longer exists
Kept 4 mip level file(s).
```

or `No orphan file: nothing was deleted.`, then `Links, not entered:` with
each link when there is one. A file that could not be judged or deleted, and
a path in no workspace, is a warning on standard error, and the command then
exits with an error.

While a run goes, a line `  ... N done in the current step` every ten
seconds counts the mipmaps made, the photographs checked or the files judged.
Ctrl+C stops the run within one stage; what it made or deleted before it
stopped stays, and the command says so.

## Interface

The command is `mipmaps` in
[`_commands/ws.py`](../../../src/sfmtool/_commands/ws.py), which expands the
paths and calls the bindings of
[`sfmtool.photographs`](../../python-bindings.md#photographs):
`PhotographCache.build_mipmaps` for a build, `check_mipmaps` and a dry run of
`prune_mipmaps` for a check, and `prune_mipmaps` for a prune, each with a
`ProgressCounter`, and prints their reports through `build_lines`,
`check_lines` and `prune_lines`. The work is in Rust, in
[`sfmtool_core::camera::mipmap`](../../../crates/sfmtool-core/src/camera/mipmap.rs):

```rust
/// Check the mipmaps of `photographs`, writing nothing.
pub fn check_mipmaps(photographs: &[&Path], tile_size: u32, progress: &Progress<'_>)
    -> Result<MipmapCheckReport, Cancelled>;
pub struct MipmapCheckReport {
    pub current: Vec<PathBuf>, pub missing: Vec<PathBuf>,
    pub stale: Vec<(PathBuf, String)>, pub in_core: Vec<PathBuf>,
    pub failed: Vec<(PathBuf, MipmapError)>,
}

/// Delete the orphan files of the mipmap directories under `paths`.
pub fn prune_mipmaps(paths: &[&Path], options: &PruneMipmapsOptions, progress: &Progress<'_>)
    -> Result<MipmapPruneReport, Cancelled>;
pub struct PruneMipmapsOptions { pub dry_run: bool, pub tile_size: u32 }
pub struct MipmapPruneReport {
    pub orphans: Vec<(PathBuf, String)>, pub kept: Vec<PathBuf>,
    pub other_files: Vec<PathBuf>, pub links: Vec<PathBuf>,
    pub failed: Vec<(PathBuf, String)>, pub outside_workspaces: Vec<PathBuf>,
}
```

```rust
use sfmtool_core::camera::mipmap::{prune_mipmaps, PruneMipmapsOptions};

let options = PruneMipmapsOptions { dry_run: true, ..PruneMipmapsOptions::default() };
let report = prune_mipmaps(&[workspace.as_path()], &options, &progress)?;
for (path, why) in &report.orphans {
    println!("{}: {why}", path.display());
}
```

[check.rs](../../../crates/sfmtool-core/src/camera/mipmap/check.rs) and
[prune.rs](../../../crates/sfmtool-core/src/camera/mipmap/prune.rs) hold
them. Both take a `&Progress`, since a workspace of hundreds of photographs
takes seconds to check: `check_mipmaps` counts photographs as `images` on
the rayon pool, `prune_mipmaps` counts the files it judges as `files`, and
each is asked between items whether to stop. Neither panics on a path:
every file that cannot be read is in the report's `failed`. The tile size is
an argument because the number of stored levels depends on it; the bindings
pass `PHOTOGRAPH_TILE_SIZE`, and the Rust tests a smaller one. A prune's
report is sorted by path; a check's lists keep the order of the photographs
given. A dry run of the prune is a field of its options, as `sfm ws upgrade`'s
is, so the check lists exactly what a prune of the same paths would delete.

## Testing

[check/tests.rs](../../../crates/sfmtool-core/src/camera/mipmap/check/tests.rs)
checks a current, a missing, a stale and an in-core photograph and a deleted one in
one call, writing nothing; a mipmap made at another quality as missing and a
workspace without the object as in core; and a cancelled check.
[prune/tests.rs](../../../crates/sfmtool-core/src/camera/mipmap/prune/tests.rs)
builds a workspace holding every kind of orphan (a deleted photograph's
files, a changed photograph's, another quality, a level past the coarsest, a
`-mip0` beside a region-readable original, an in-core photograph's, an old
temporary file) beside current files, a stopped build's current finer files,
a new temporary file, files whose names are not a mip level file's, and a file
named like one outside the mipmap directory; a prune deletes exactly the
orphans and a dry run deletes nothing. Further tests: every mip level file of
a workspace without the object is deleted; a path searches only its mirror, a
directory's and a deleted photograph's; a path in no workspace; an invalid
object searches nothing; a nested workspace's files; a junction (Windows) or
symbolic link (elsewhere) in a mipmap directory, or a mipmap directory that is
one, is not entered and what it leads to stays; a symbolic link to a file is
not deleted (Unix); the last check before a delete refuses a file whose
directory is outside the mipmap directory; and the names read as mip level or
temporary files.
[test_mipmap_check_prune_rust_bindings.py](../../../tests/rust_bindings/photographs/test_mipmap_check_prune_rust_bindings.py)
checks the bindings' reports and counters, and
[test_ws_mipmaps.py](../../../tests/test_ws_mipmaps.py) the command's
output, exit status and files through a check, a build, a check, and a
prune after the photograph is deleted, in a workspace with the object and
without it.
