# Out-of-core photograph pyramid

**Status:** Draft

Decided by the user, and followed here: level 0 of a photograph's pyramid is
always the original image file and is never stored again; the reduced levels of
a large photograph are JPEG files in `.sfm-cache/` at the workspace root, one
file per level, in directories that mirror the workspace; once a photograph has
these files every reader, the display and the patch kernels alike, reads its
reduced levels from them; the unit of display and of region reads is a 512 px
square; and there is no stopgap fix. Open: the questions at the end, each with
a recommendation. Building only the part of each level that one patch's render
can reach is a separate, optional draft,
[view-region-pyramid.md](view-region-pyramid.md), which this design does not
depend on.

Amends [core/camera/photograph-cache.md](../core/camera/photograph-cache.md)
(the cache gains photograph tiles and an on-disk tier, and its pyramids a depth
set by image size),
[core/camera/reading-and-writing-images.md](../core/camera/reading-and-writing-images.md)
(a second JPEG writer and reader, for mipmaps),
[gui/multi-panel-image-browser.md](../gui/multi-panel-image-browser.md)
§ "Image Detail" (the panel draws photograph tiles),
[gui/mcp-server.md](../gui/mcp-server.md) § "The Image Detail view" (two
fields), and
[workspace/workspace.md](../workspace/workspace.md) § "Workspace Directory
Layout" (the cache directory). The first two link back here.

SfM Explorer shows a photograph, and the patch kernels sample it, through one
image pyramid per photograph: the photograph and successively halved copies of
it. Today each pyramid is built whole in memory from one decode of the file,
and Image Detail uploads level 0 as a single GPU texture. That works for the
test data and for 4K video frames, but a 200-megapixel phone photograph
(16320 x 12240) is 571 MiB as RGB and 762 MiB as a pyramid, and wider than the
8192 px texture the viewer's GPU device allows, so selecting one crashes the
viewer. This draft makes the size of a photograph matter only to how long its
first read takes: the display draws 512 px tiles of the pyramid level that
matches its zoom, so it never makes a texture larger than a tile and shows the
original pixels when zoomed in far enough; and the reduced levels of a large
photograph are computed once, stored as JPEG files beside the workspace, and
read a tile at a time, so that a session that looks at a photograph decodes only
what it shows, and every later session and every kernel reads the same levels.

## The problem, measured

Everything below was measured on one of the user's photographs,
`20261115_164358.jpg` (16320 x 12240, baseline JPEG with 4:2:0 chroma and a
restart interval of 4 MCUs, 29,995,237 bytes), read only, on the development
machine (Intel i9-14900HX, 32 threads, 64 GB). The machine was shared with
other work, so each time is given as the range over the runs; the low end is
the unloaded figure. The program is a scratch Rust crate against the workspace's
pinned `jpeg-decoder` 0.3.2 and `jpeg-encoder` 0.7.1 (with its `simd` feature,
as the workspace builds it); it is described under "Measurements" so it can be
written again.

| Quantity | Value |
|----------|-------|
| decode of the original with `jpeg-decoder` (`ImageU8::read_rgb`'s JPEG path) | 0.79 to 1.7 s |
| decoded size, RGB | 571.5 MiB |
| levels 1 to 6 by `downsample_2x` | 0.18 to 0.37 s; 142.9, 35.7, 8.9, 2.2, 0.56, 0.14 MiB |
| level 5 | 510 x 382 (12240 / 32 is 382.5, and `downsample_2x` truncates) |
| restart intervals in the file | 195,075 (1020 x 765 MCUs of 16 x 16 px, 4 MCUs each) |
| the same restart interval in the other 16 photographs of the capture | 4 MCUs in every one |
| a pyramid of 6 levels (`DISPLAY_PYRAMID_LEVELS`) | 798.8 MB (761.8 MiB) |

The viewer asks for `wgpu::Limits::default()` (`adapter.request_device(&wgpu::DeviceDescriptor::default())`
in [lib.rs](../../crates/sfm-explorer/src/lib.rs)), whose
`max_texture_dimension_2d` is 8192, and passes that limit to egui as its
`max_texture_side`. Two readers make one texture of a whole photograph:

- Image Detail, in `load_image` in
  [image_detail/mod.rs](../../crates/sfm-explorer/src/image_detail/mod.rs),
  expands level 0 to an RGBA `ColorImage` on the GUI thread (a second full-size
  copy, 762 MiB for this photograph) and calls `ctx.load_texture`.
- The 3D view's camera-view background, `load_bg_texture` in
  [bg_image.rs](../../crates/sfm-explorer/src/scene_renderer/upload/bg_image.rs),
  reads the photograph with `ImageU8::read_rgba` on the GUI thread, outside the
  photograph cache, and makes a texture of its full size with one mip level. A
  photograph it cannot read is read again on the next frame.

wgpu reports a texture wider than the limit as a validation error, and its
default handler for one is to panic, so either reader ends the viewer
(`Dimension X value 16320 exceeds the limit of 8192`). The thumbnail and patch
atlases already page against the limit
([thumbnails.rs](../../crates/sfm-explorer/src/scene_renderer/upload/thumbnails.rs)).

Memory is the second problem. The photograph cache's default budget is a
quarter of physical memory clamped to 1 to 16 GiB, so on the 64 GB machine it
holds about 21 of these pyramids and on a 32 GB machine 10. Track View asks
for the pyramid of every image of the viewed track on every frame
(`cache_selected_point_images` and `cache_bench_track_images` in
[dock.rs](../../crates/sfm-explorer/src/dock.rs)); with 17 images that is
13.6 GB, so below that budget each landing decode evicts another that the next
frame asks for again, and the decodes never end.

## Terms

These are proposed for [GLOSSARY.md](../GLOSSARY.md), each with its reason.

| Word | Means | Not | Why |
|------|-------|-----|-----|
| **mipmap** | one JPEG file holding one reduced level of a photograph's pyramid, `<base>-mipmap<f>.jpg` under `.sfm-cache/`, where `f = 2^level` is the factor the level is reduced by; `sfmtool_core::camera::mipmap` | `level file`, `cache file`, `thumbnail`, `pyramid file` | the word the user chose for the files, and the suffix the file names carry; it is the graphics word for a stored reduced copy at a power-of-two factor. *Level* alone is the in-memory level, which for a small photograph has no file |
| **cache directory** | `.sfm-cache/` at a workspace's root, holding files derived from the workspace's images that can be deleted at any time and rebuilt; today only mipmaps | `cache folder`, `.cache` | it names what is in it (files that are a cache) and where (the workspace's); a name without `sfm` would collide with other tools' caches |
| **out-of-core photograph** / **in-core photograph** | a photograph whose width or height exceeds `OUT_OF_CORE_LONG_SIDE` (4096) is out of core (with one exception, § "Which photographs are out of core"): its reduced levels come from mipmaps and the display reads it a tile at a time; any other photograph is in core: its whole pyramid is built in memory from one decode, as every pyramid is today | `large image`, `tiled image`, `big photograph` | the tier decides where the levels live, which is what *core* means here; *large* says nothing about what is done differently, and every photograph is shown in tiles |
| **photograph tile** | a 512 x 512 block of one level of a photograph's pyramid, with a one-pixel border of its neighbours' pixels: the unit the display draws as one texture and that a level is read in; `PhotographTile`, keyed by level, column and row | bare `tile`, `block`, `chunk` | bare *tile* is a view's tile, the patch a kernel renders ([GLOSSARY.md](../GLOSSARY.md) § "Rendering a view's tile"), and Track View draws those beside photographs, so the photograph's blocks need a qualified name |
| **level builder** | the viewer's worker that writes the missing mipmaps of the open reconstructions' out-of-core photographs, in the order of what is on screen | `mipmap generator`, `preprocessor` | it names what it does, build levels; the panel line says *Building image levels* because a person thinks of an image's levels, not of the files they are kept in |

## Which photographs are out of core

A photograph is **out of core** when its width or height, read from its header
with `image_dimensions`, is greater than `OUT_OF_CORE_LONG_SIDE = 4096`; it is
**in core** otherwise. One more photograph stays in core: one whose level 1
would be wider or taller than 65535 px, the most a JPEG frame can hold, which
needs a side over 131071 px (a strip such as 140000 x 3000 still fits the
reader's 4 GiB decode limit). The test data (270 x 480 to 2040 x 1536 and 480 x 480),
3840 x 2160 video frames and 4032 x 3024 phone photographs are in core and keep
today's in-memory pyramids, so nothing is written for them and no result
changes except through the pyramid depth (below). A 24-megapixel camera
(6000 x 4000) and the 200-megapixel phone photographs are out of core.

Why 4096 rather than 8192, the texture limit: once every photograph is drawn in
tiles (below), the texture limit no longer separates anything, and the tier is
about memory and the time of a first read. A 6000 x 4000 photograph is 72 MB
at level 0 and 96 MB as a pyramid, so a few hundred of them, a normal camera
capture, do not fit any budget, while their mipmaps cost one extra encode each.
4096 is also where `sfm ws init` stops SIFT image doubling
(`double_image_below_long_side`), so a photograph in core is one SIFT may
double. The threshold is a constant, not a setting, because which tier a
photograph is in decides which pixels the kernels read; a change to it is a
change of results and is made as one. Tests pass a smaller threshold through
`MipmapOptions` (below), which production code does not.

## Levels by size

Two counts are decided by the photograph's size, not fixed.

**Pyramid depth.** Every photograph pyramid has
`ImageU8Pyramid::full_levels(width, height)` levels, down to one or two pixels
on the short side: 14 levels for 16320 x 12240, 12 for 3840 x 2160, 9 for
270 x 480. This is what `cluster-patches`, the viewer's cluster-patches build
and the Python patch bindings already build, so the viewer, `sfm web-export`
and the CLI patch commands read the same depth, and the cluster-patches build
can share the viewer's cache. The viewer's fixed 6 (`DISPLAY_PYRAMID_LEVELS`,
which `PhotographCache::new` takes today) goes away. The extra levels cost
under 0.1% of a pyramid's bytes.

**Mipmap count.** An out-of-core photograph has mipmaps for levels 1 to `K`,
where `K` is the first level whose longer side is at most one tile (512 px), or
the last level whose sides are both at least 1 px if that comes first. Levels
below `K` are not stored: they are box-filtered in memory from the decoded
level `K`, which is deterministic because its input is the file. So the
coarsest stored level of every out-of-core photograph fits in one photograph
tile.

| Photograph | Levels | Mipmaps (`K`) | Coarsest mipmap |
|------------|--------|---------------|-----------------|
| 16320 x 12240 | 14 | 5: `-mipmap2` to `-mipmap32` | 510 x 382 |
| 6000 x 4000 | 12 | 4: `-mipmap2` to `-mipmap16` | 375 x 250 |
| 4097 x 2304 | 12 | 3: `-mipmap2` to `-mipmap8` | 512 x 288 |
| 30000 x 1000 (a panorama) | 10 | 6 | 468 x 15 |

The third row shows the rule's edge: level 3 of a 4097 px side is 512 px, which
is one tile, so `K = 3`. Level `l` is `floor(width / 2^l)` by
`floor(height / 2^l)`, as repeated `downsample_2x` makes it, and its pixel `j`
covers level-0 pixels `j·2^l` to `(j+1)·2^l − 1`; the level-0 columns past
`w_l · 2^l` (fewer than `2^l`) are dropped, as today.

**What changes for the kernels.** Where a kernel clamps its level at the
pyramid's top today, a deeper pyramid lets it read a coarser level. With 6
levels the top is level 5, so the result changes only for samples that ask for
more than level 5:

- `BilinearMip` (`mip_level_for_sigma` in
  [remap.rs](../../crates/sfmtool-core/src/camera/remap.rs)): where
  `σ_major ≥ 2^5.5 ≈ 45.3` photograph pixels per output pixel.
- `Anisotropic` (`aniso_footprint`): where `σ_minor > 32`, which today reads
  level 5 alone and now blends levels 5 and 6.
- The cluster refinement's per-grid level (`level_for_map` in
  [cluster_refine.rs](../../crates/sfmtool-core/src/patch/cluster_refine.rs)):
  where the smaller singular value `s_min ≥ 64`, in Track View's cluster-stage
  tiles (`sample_member_grid` on the viewer's pyramids). `cluster-patches` and
  the viewer's cluster-patches build already read full depth and do not change.
- Track View's crops (`patch_pictures/crop.rs`, `floor(log2 step)` clamped to
  the top level): where the step is 64 or more.

These are views that shrink a patch more than 32 to 45 times, which happen on
small patches seen from far away, in every photograph size including the test
data. In-core photographs change nowhere else.

## Mipmap files

### Where they are and what they are called

A photograph's mipmaps are found from its path alone, so the cache's interface
keeps taking a path. The **workspace root** is the closest ancestor of the path
holding `.sfm-workspace.json`, found by `find_workspace_for_path`
([archive-io-crate.md](../formats/archive-io-crate.md); the walk goes up the
path as given and does not resolve links, so an image folder that links to
storage elsewhere still belongs to the workspace around the link). The
photograph's **relative path** is its path relative to that root. The mipmap of
level `l` is

```
<root>/.sfm-cache/<parent of relative path>/<base>-mipmap<2^l>.jpg
```

for example `<ws>/images/IMG_0001.JPG` has
`<ws>/.sfm-cache/images/IMG_0001-mipmap2.jpg` to
`<ws>/.sfm-cache/images/IMG_0001-mipmap32.jpg`. The parts:

- **`<base>`** is the file name without its last extension, as Rust's
  `Path::file_stem` gives it, kept in its own case: `a.b.png` gives `a.b`,
  `IMG_0001.JPG` gives `IMG_0001`, a name with no extension keeps the whole
  name, and a name that starts with a dot and has no other (`.x`) keeps `.x`.
- **The extension is always `.jpg`**, in lower case, whatever the original's
  format.
- **The directory parts are the relative path's own**, in their own case, with
  `/` between them.

That name is not one-to-one: `x.jpg` and `x.png` in one directory, or `x.jpg`
and `x.jpeg`, have the same `<base>`. Every mipmap therefore records the
relative path of the photograph it was made from (below), and a reader that
finds a mipmap recording another photograph does not use it. A photograph
whose first-form name is taken that way uses the **second form**,
`<file name>-mipmap<2^l>.jpg`, its whole file name with its extension
(`x.png-mipmap2.jpg`). If the second form is taken too (a photograph named
`x.png.jpg` beside `x.png`), the photograph gets no files and is read through
the in-memory equivalent (below), with a warning naming both photographs.
Which of two colliding photographs takes the first form depends on which was
built first, which is why the record and not the name is what a reader trusts.
The open questions ask whether to use the second form for every photograph.

On a case-insensitive file system two relative paths that differ only in case
are one file, so they cannot both exist, and their mipmaps cannot either. A
workspace made on a case-sensitive file system with `X.jpg` and `x.jpg` and
copied to a case-insensitive one collides in the cache directory; the records
catch it and the two photographs fall to the second form and, colliding again,
to the in-memory equivalent.

A photograph gets no files, and is read through the in-memory equivalent, when
no workspace root is found above it, when its relative path has a `..`
component (an image name such as `../other/x.jpg` is accepted in a
reconstruction's image list and would put the mirror outside the cache
directory), or when the cache directory cannot be written.

### What is written in one

A mipmap is a baseline JPEG that any program can open:

| Setting | Value | Why |
|---------|-------|-----|
| encoder | the `jpeg-encoder` crate, as `ImageU8::write` uses | one JPEG encoder in the tree |
| quality | 95 (`MIPMAP_JPEG_QUALITY`, equal to `DEFAULT_JPEG_QUALITY`) | at level 1 of the measured photograph the mean absolute error is 0.49 grey levels at 95 and 0.73 at 90, for 10.8 MB against 7.3 MB; the kernels read these pixels, so the lower error is worth the bytes |
| chroma | 4:4:4 (`SamplingFactor::F_1_1`) for colour; one component (`ColorType::Luma`) when every pixel of level 0 has R = G = B | a tile can be decoded exactly only without chroma subsampling (below); 4:2:0 would save 14% (9.2 MB against 10.8 MB at level 1) |
| restart interval | 16 MCUs (`MIPMAP_RESTART_INTERVAL_MCUS`), 128 px of one MCU row at 4:4:4 | the unit a tile is decoded in (below); 2.4% larger than no restart markers (10.77 MB against 10.52 MB at level 1, quality 95) |
| Huffman tables | the standard ones, not optimized | the encoder writes optimized tables as one scan per component, which a tile cannot be cut from |
| metadata | one APP15 segment, the record below; no JFIF density, no EXIF, no orientation | the record is how a reader knows the file is current |

All five mipmaps of the measured photograph at quality 95 are 15.2 MB together
(10.77, 3.21, 0.90, 0.25, 0.07 MB), half the original's 30.0 MB; at quality 90
they are 10.2 MB.

**The record** is one APP15 segment: the ASCII bytes `sfmtool-mipmap`, a zero
byte, then a JSON object in UTF-8, under 1 KB:

```json
{
  "version": 1,
  "source": "images/IMG_0001.JPG",
  "source_size": 29995237,
  "source_modified_ns": 1763255640000000000,
  "source_xxh128": "3a9f…",
  "source_width": 16320,
  "source_height": 12240,
  "level": 1,
  "width": 8160,
  "height": 6120,
  "stored_levels": 5,
  "jpeg_quality": 95,
  "components": 3,
  "restart_interval_mcus": 16,
  "encoder": "jpeg-encoder 0.7.1",
  "sfmtool_version": "…"
}
```

`source` is the relative path with `/` separators. `source_xxh128` is the
XXH128 of the original's bytes, the hash a `.sift` file records as
`image_file_xxh128` (`xxh128_of_file` in
[sift/files.rs](../../crates/sfmtool-core/src/features/sift/files.rs)). The
record lives in each file rather than in a manifest beside them, so a file
cannot be separated from the facts about it, and two writers never edit one
shared file.

### How the levels are made

From one decode of the original, held in memory:

1. Decode the original with `ImageU8::read_rgb`, the decoder every reader
   uses; this is level 0 as the kernels see it.
2. Make levels 1 to `K` by `downsample_2x`, each from the previous **exact**
   level in memory, as `ImageU8Pyramid::from_image` does.
3. Encode each of levels 1 to `K` on its own, coarsest first, with the settings
   above.
4. Write each through a temporary file renamed into place (below).

The alternative is to make level `l + 1` from the decoded JPEG of level `l`.
Making each level from the exact level above it is chosen because:

- **Each file carries one JPEG loss, not a chain of them.** Level 3 from
  decoded files would carry the losses of levels 1, 2 and 3 filtered together.
- **Any level can be made again alone** from the original, to the same bytes
  (with the same build, below), so a lost or damaged file is replaced without
  touching the others. A chain would make every coarser level depend on the
  exact bytes of every finer file.
- **It costs nothing extra.** Level 1 needs the whole decode either way; the
  exact levels below it cost 0.18 to 0.37 s for all of them.

The levels are encoded coarsest first so that the coarsest, which is what a
panel shows first, is ready about a second after the decode starts rather than
after the whole set.

**Determinism.** The same original, decoded by the same build of the same
decoder, gives the same level 0; `downsample_2x` is integer arithmetic; and
`jpeg-encoder` with fixed settings writes the same bytes for the same pixels on
one machine. Whether its AVX2 path and its scalar path write the same bytes is
not established, and neither is whether `jpeg-decoder`'s SSSE3 inverse DCT
and its scalar one agree bit for bit (the decoder picks them at run time). This
does not matter for consistency inside a workspace, because **the files are the
record**: once a mipmap exists, every reader reads it, whatever build wrote it.
A newer build never rewrites a valid mipmap to its own bytes, since that would
change results without any input changing. The originals have the same
property today: a JPEG decoded on two machines can differ by the decoder's
platform paths.

Peak memory while one 200-megapixel photograph is made is about 1 GB: 571 MiB
for the decode, the decoder's own component planes while it decodes (about
290 MiB for 4:2:0 at this size), 190 MiB of reduced levels and 15 MB of encoded
files. This is an estimate from the buffer sizes, not a measurement; step 4
measures it. The time is the decode (0.79 to 1.7 s), the reduction (0.18 to
0.37 s) and the encode of all levels at quality 95 (1.2 and 2.8 s in two runs
under load; about 0.6 s unloaded, estimated from level 1's 0.47 s in the
quietest run and the other levels' third of its pixels), so about 1.6 s
unloaded, estimated.

### The in-memory equivalent

When a photograph's mipmaps cannot be written (no workspace root, a `..` in the
relative path, a collision of both name forms, a read-only cache directory, a
full disk), a reader still needs the pixels the files would hold, because the
kernels must not read the exact box-filtered levels for one photograph and the
JPEG ones for another. It makes the same files in memory instead: steps 1 to 3
above, then decodes the encoded bytes instead of writing them. With the same
build these are the pixels the files would have held. It costs the whole making
of the set in every session, and a warning in the log says why the files were
not written. The same path serves the session in which a set is first made: the
reader decodes the bytes it has just encoded rather than reading back the files
it has just written.

### When a mipmap is current

A mipmap is **current** when all of these hold, checked in order, and **stale**
naming the first that fails otherwise:

1. It reads as a JPEG with a record of a known `version`, its frame is
   `width` x `height`, and it has a restart interval.
2. Its `source` is the photograph's relative path, its `level` is the level
   asked for, and its `width` and `height` are that level's size for the
   original's size.
3. The original's header size equals `source_width` x `source_height`.
4. The original's length equals `source_size` and its modification time equals
   `source_modified_ns`. When the length matches and the time does not (a copy,
   a `git checkout`), the original is hashed and compared with
   `source_xxh128`; equal hashes make the mipmap current, and the reader
   remembers for the session that this original's time differs, so it hashes it
   once per session, not on every read. A different length is stale without a
   hash.

A mipmap that does not read, is cut short, or fails any test is treated as
missing, and the photograph's whole set is made again: making one level costs
the whole decode anyway, and the other levels add only their encodes. It is
never a panic and never an error to the caller unless the original itself
cannot be read. A reader checks the coarsest mipmap when it first needs
the photograph, and each finer one when it first opens it, so a session's start
costs one small file per out-of-core photograph and one `stat` of each
original.

### Writers that meet

Every write goes through a temporary file in the same directory
(`.<file name>.<process id>-<count>.tmp`), renamed over the target, as
`ImageU8::write` already does. Unlike `ImageU8::write`, a mipmap writer whose
rename fails does **not** write the target directly: if the target now exists
and is current, another writer has made it, and the write counts as done;
otherwise the photograph falls to the in-memory equivalent for this session.
The temporary file is removed either way.

Two writers of one photograph in one process are one: the cache makes each
photograph's set once, and every other asker waits for it. Two processes (two
viewers, or `sfm sift` and a viewer) can make the same set at the same time;
each writes the same pixels, and the last rename wins. A reader reads a whole
mipmap into memory when it opens it (at most 10.8 MB for this photograph) and
keeps those bytes, so a session sees one version of each file even when another
process replaces it, and holds a file open only while it reads it. Rust's
standard library opens files on Windows with delete sharing, so an open
mipmap should not block another process's rename over it; the two-process test
in step 4 confirms this on Windows, and the rule above (a failed rename over a
current file counts as done) covers it if it does not.

### Keeping the cache directory out of the way

- On creating `.sfm-cache/`, the writer puts in it a `.gitignore` holding `*`
  and a `CACHEDIR.TAG` with the standard signature line
  (`Signature: 8a477f597d28d172789f06886806bc55`), which backup tools skip.
  No `.gitignore` of the user's is edited.
- **Image discovery must skip it.** `expand_paths` in
  [_filenames.py](../../src/sfmtool/_filenames.py) expands a directory with
  `rglob("*")`, so `sfm sift --extract .` at a workspace root, or a `solve` or
  `pick-images` over it, would take every mipmap for an image. The same holds
  for `_sensor_images` in
  [camrig/from_rig_config.py](../../src/sfmtool/camrig/from_rig_config.py) and
  `find_images` in [camrig/create.py](../../src/sfmtool/camrig/create.py), whose
  `**` pattern can reach it. Each skips any path under a directory named
  `.sfm-cache`. This lands before anything writes the directory (step 4).
- **Orphans.** A mipmap whose original is gone, or whose record no longer
  matches, is left where it is by every reader: deleting files is not a
  reader's business, and a stale file is rewritten when its photograph is next
  made. `sfm ws mipmaps --prune` (below) deletes the mipmaps whose record names
  a photograph that no longer exists or that they are stale for, and temporary
  files older than an hour. Deleting `.sfm-cache/` by hand is always safe.
- `test-data` is never written: every image in it is in core.

### Is it a file format?

The [I/O-shape standard](../STANDARDS.md#every-file-format-has-the-same-io-shape)
is about the formats sfmtool defines, one per format crate. A mipmap is a plain
JPEG with one extra segment, read by its own code in `sfmtool_core::camera`
rather than through `sfmtool.fileio`, and nothing outside sfmtool reads the
record, so it is not a format in that sense: no format crate and no
`read_mipmap` / `write_mipmap` / `verify_mipmap` quartet. The record's fields,
the naming rule and the encoder settings are specified in this design and, once
filed, in a section of the photograph-cache spec, which is the one place that
defines them. If a later tool wants to read mipmaps (the web export, say), the
record becomes a format and gets a spec under `specs/formats/`.

## Reading pixels

### A tile of a mipmap

A photograph tile of level `l ≥ 1` is decoded from the level's mipmap without
decoding the rest of it. The method relies on the restart markers the writer
put in:

1. **Index the file once** when it is opened: scan its entropy-coded data for
   the markers `0xFFD0` to `0xFFD7` (stepping over the stuffed `0xFF00`) and
   record the byte range of each restart interval. The count must equal the
   level's MCUs divided by the interval, rounded up, and the markers must cycle
   0 to 7; otherwise the file is stale (corrupt). Indexing level 1 of the
   measured photograph (48,769 intervals) took 2 to 7 ms.
2. **Find the intervals the rectangle needs.** With `M` MCUs per row and `N`
   MCUs per interval, the MCU at row `r`, column `c` is in interval
   `(r·M + c) div N`. The tile with its one-pixel border needs, in each MCU row
   it spans, the MCUs that cover its columns; the intervals holding them are
   the sorted union.
3. **Assemble a small JPEG** from the file's own segments up to and including
   its start of scan (quantization and Huffman tables, restart interval),
   with the frame's size changed to `N·8` wide and `8·k` high for `k`
   intervals, followed by the `k` intervals' bytes with restart markers
   numbered 0 to 7 again between them, and an end-of-image marker. Each
   interval starts with the DC predictions reset, so the intervals decode on
   their own, and each becomes one row of MCUs of the small frame, wherever it
   lies in the level (an interval may run from the end of one MCU row into the
   start of the next). The level's last interval holds fewer than `N` MCUs when
   the level's MCU count is not a multiple of `N`; it is decoded in a frame of
   its own, as wide as it is.
4. **Decode it with `jpeg-decoder` set to `ColorTransform::None`**, which
   returns the Y, Cb and Cr samples as decoded, and convert them to RGB with
   one fixed formula: the fixed-point BT.601 conversion `jpeg-decoder` uses on
   its scalar path (`ycbcr_to_rgb`, from libjpeg-turbo's `jdcolext.c`). Then
   copy each MCU's 8 x 8 pixels to its place in the level and crop the tile.

Measured on level 1 of the photograph (8160 x 6120, quality 90, interval 16
MCUs), nine tiles including the right and bottom edges: 2.3 to 6 ms per tile
on an unloaded machine (up to 15 ms under load), each identical to the same
pixels of a decode of the whole file through the same raw path.

Two choices are needed for that identity:

- **The raw path.** Decoded to RGB by `jpeg-decoder` itself, the tiles
  differed from the whole decode by up to 1 grey level. Its SSSE3 colour
  conversion (`color_convert_line_ycbcr` in its `arch/ssse3.rs`) converts all
  but the last 8 to 15 pixels of each line and leaves those to the scalar
  formula, which rounds differently; the small frame's lines end at different
  places from the level's. Converting outside the decoder with one formula makes
  every pixel's value depend on its decoded samples alone.
- **4:4:4.** With 4:2:0 chroma, `jpeg-decoder` upsamples the chroma with the
  triangle filter libjpeg calls fancy upsampling (`UpsamplerH2V2` in its
  `upsampler.rs`), which reads the chroma samples of the neighbouring MCUs. In
  the small frame those neighbours are other intervals, so the pixels at
  interval edges would differ. A 4:4:4 file has no upsampling, and an 8 x 8
  block's pixels depend only on its own coefficients.

**A whole level** is decoded from the whole file through the same raw path and
conversion, which is what the identity was measured against. It is what the
kernel pyramids read (below). A whole decode of level 1 of this photograph to
RGB took 0.16 to 0.45 s; the raw path does the same work less the colour
conversion, which it then does itself.

### A region of level 0

Level 0 is the original, in any format the reader reads. Two ways to get a
region of it:

- **The whole decode**, always available: `ImageU8::read_rgb`, 0.79 to 1.7 s
  and 571 MiB for this photograph, cut into tiles. The cache keeps the decoded
  level 0 as one entry while tiles of it are being asked for (below), or cuts
  them from the photograph's pyramid entry when one is held, so
  panning at full resolution does not decode the file again for each tile.
- **The restart-interval fast path**, for an original JPEG that allows it:
  decode only the restart intervals the region needs. It must give exactly the
  pixels of the whole decode, because a panel and a kernel can read the same
  pixel through either. An original qualifies when it is a sequential Huffman
  JPEG (baseline or extended, `SOF0` or `SOF1`) of 8-bit samples with one scan
  holding every component; it has one component, or three with Cb and Cr at
  1 x 1 and Y at 1 x 1, 2 x 1, 1 x 2 or 2 x 2; it has a restart interval `R`
  that divides its MCUs per row, so every interval lies within one MCU row and
  every row starts one; and indexing finds every interval, in order. Every
  other original (progressive, arithmetic-coded, no restart interval, or one
  that does not divide the row, PNG, TIFF, 16-bit, any other format) uses the
  whole decode.

  The small JPEG keeps the original's layout in two dimensions: its columns
  are the region's MCU columns widened to interval boundaries, its rows are
  the region's MCU rows, both with a margin, and its intervals are copied in
  raster order. It is decoded to RGB by the same `jpeg-decoder` path
  `read_rgb` uses, then cropped. The margins make the cropped pixels equal the
  whole decode's: one MCU above, below and to the left, for the chroma
  upsampling's neighbours; and on the right at least one MCU and at least 16
  px, so the cropped pixels are never among the last 8 to 15 of a line, which
  the decoder converts with its scalar formula. Where the region reaches the
  image's right or bottom edge, the small frame's width or height is set to
  the distance to that edge, so the edge is padded as in the whole decode, and
  the scalar tail falls on the same pixels in both.

  Measured on the photograph (4:2:0, interval 4 MCUs, one interval per 64 px):
  five 512 px regions, among them the top-left corner and the bottom-right
  corner, 2.6 to 3.5 ms each, reading about 29 KB of the file each, every pixel
  equal to the whole decode. Indexing the original's 195,075 intervals took 20
  to 63 ms.

  This depends on how `jpeg-decoder` 0.3.2 upsamples and where its SIMD
  conversion stops, so it is guarded by a test that compares every region of
  a set of small JPEGs with the whole decode (step 9), and the reading spec
  says the reader may change decoder: the fast path goes with the test, and a
  decoder that fails it turns the fast path off rather than changing pixels.

## The caches in memory

`PhotographCache` keeps its job and its key, and gains a second kind of entry.

**Pyramids** (the kernels' entries, as today). `get`, `peek`, `peek_state` and
`get_many` answer with an `Arc<ImageU8Pyramid>` of the photograph's whole
depth. For an in-core photograph it is built as today. For an out-of-core one,
level 0 is `read_rgb` of the original, levels 1 to `K` are the mipmaps decoded
whole through the raw path, and the levels below `K` are `downsample_2x` of
decoded level `K`. When the mipmaps are missing or stale, the `get` that finds
it makes them first (and is the asker that writes them; the level builder and
any other asker wait on the same slot), so a kernel never reads levels that are
not the files' pixels. Bytes and time are about today's: 762 MiB, and the
decode plus 0.2 to 0.5 s of mipmap decodes (or the making of the set, once).
The pyramid budget and its eviction are unchanged.

**Photograph tiles** (the display's entries, new). Keyed by path, level,
column and row; each holds the 512 x 512 core and a one-pixel border taken from
the neighbouring tiles' pixels, or the edge pixel repeated at the image's edge
(514 x 514 x 3 = 793 KB at most). The border is what lets the display draw
tiles side by side with linear filtering and no seam. Where a tile comes from:

| Photograph | Level | Source |
|------------|-------|--------|
| in core | any | cut from the photograph's pyramid entry, which a tile request asks for |
| out of core | 1 to `K` | the level's mipmap, a tile decode from its bytes, which the entry keeps once read (10.8 MB for level 1 here) |
| out of core | below `K` | the levels computed from level `K`, kept whole with the coarsest |
| out of core | 0 | the fast path from the original's bytes, which the entry keeps while level 0 is read (30 to 40 MB); otherwise the whole decode, kept as one entry |

The tile entries have a budget of their own, so a kernel's pyramids cannot
evict what the display is showing, and the other way round: by default a
sixteenth of physical memory clamped to 512 MiB to 4 GiB, overridden by
`SFM_EXPLORER_TILE_CACHE_MB` as the pyramid budget is by
`SFM_EXPLORER_PHOTOGRAPH_CACHE_MB`. Eviction is least recently used, with two
exceptions: the coarsest level of every out-of-core photograph whose mipmaps
have been read, with the levels below it (at most one tile, 0.78 MB for this
photograph), is evicted only after every other tile entry; and a whole level-0
decode is evicted before any tile, since it is 571 MiB and is made again by
the same decode.

The contract stays the one the viewer is built on: `peek_tile` never reads or
waits, `get_tile` decodes or waits, and a remembered failure is not tried again
until the file's stamp changes. A tile decode uses no rayon, so a tile asked
for inside a rayon loop cannot deadlock on a slot, for the reason
[photograph-cache.md](../core/camera/photograph-cache.md) § "Implementation
notes" gives; making a set of mipmaps uses no rayon either.

**Memory, two cases.**

- *17 photographs of 16320 x 12240.* The display holds the visible tiles of one
  level: at about one texel per screen pixel a 3840 x 2160 panel covers at most
  9 x 6 = 54 tiles, 43 MB, plus the coarsest resident of each photograph,
  13 MB, plus the mipmap bytes of what it has opened (15 MB per photograph at
  most). A kernel that reads all 17 (opening a file whose patch bitmaps must be
  rendered, `get_many`) holds 17 pyramids of 762 MiB, 12.6 GiB, as it would
  today; that is what [view-region-pyramid.md](view-region-pyramid.md)
  removes.
- *368 frames of 3840 x 2160.* In core, as today: 33.2 MB per pyramid, 12.2 GB
  for all of them, bounded by the pyramid budget. The display cuts its tiles
  from those pyramids; the coarsest level it keeps of each frame it has shown
  is 480 x 270, 389 KB, 143 MB for all 368.

**Track View's requests.** Until the view-region pyramid lands, Track View
reads whole pyramids, and with out-of-core photographs the endless
decode-and-evict loop above is real. The panels' requests made in one frame are
therefore admitted in the order they are made only while the estimated bytes of
the pyramids they ask for (from header dimensions, levels counted) fit within
80% of the pyramid budget, counting the pyramids already in the cache that the
frame draws from; the photographs that do not fit are not asked for, and their
tiles show *Not loaded: photograph cache is full* with a hover naming
`SFM_EXPLORER_PHOTOGRAPH_CACHE_MB`.

## Who makes the mipmaps, and when

- **`sfm sift --extract`** makes them for every out-of-core photograph it
  reads, after the `.sift`, so a solve's workspace has them before the viewer
  opens it. The Rust `build_sift_file` in
  [sift/files.rs](../../crates/sfmtool-core/src/features/sift/files.rs) (the
  viewer's *Build SIFT files* and `pick-images`) makes them from the decode it
  already holds, adding only the reduction and the encode (about 0.8 s for this
  photograph, estimated from the parts above, beside an extraction of the
  same photograph that takes longer than its decode). The Python
  extractors (`sfmtool` and OpenCV backends decode with `read_image_rgb`; COLMAP
  decodes inside COLMAP) call one binding per photograph after the `.sift` is
  written, which decodes the photograph again (0.8 to 1.7 s). The open
  questions ask whether that binding should take the decoded array instead.
- **`sfm ws mipmaps PATHS…`** makes them explicitly for the photographs under
  the paths (skipping in-core ones and current sets), with `--prune` to delete
  orphans and `--check` to report stale sets without writing. It is the way to
  prepare a workspace whose features came from somewhere else.
- **The viewer's level builder.** When a reconstruction opens, the builder
  queues every out-of-core photograph of it whose coarsest mipmap is missing or
  stale (a `stat` and one small read each), and builds them on threads of its
  own, not the rayon pool and not the background queues: the selected image
  first, then the images the Image Browser strip shows, then the rest in image
  order; a photograph a panel asks for moves to the front. Its concurrency is
  the number of sets that fit in a quarter of physical memory at the estimated
  1 GB per 200 megapixels (5 bytes per source pixel), at most half the cores,
  at least one; `SFM_EXPLORER_LEVEL_BUILDS` overrides it. While it works,
  the Background Tasks panel's quiet line says *Building image levels:
  {done} of {total}*. It is not a quiet task: quiet tasks run one at a time in
  a queue and give way to asked-for tasks, while this work touches no node and
  must not wait behind a long edit. It stops when the viewer quits; a set it
  did not finish has no partial files, and is built next time.
- **A kernel** that needs an out-of-core photograph whose mipmaps are not
  current makes them itself inside its read phase (the `get` above), counted in
  the phase's note (*3 of 17 read, 2 levels made*). It does not read the exact
  levels in the meantime.
- **A panel** that needs one shows what exists: the coarsest level the moment
  its file is written (or decoded in memory), finer tiles as the set
  completes, and *Building image levels…* in place of *Loading image…* until
  the coarsest exists. The display reads the encoded bytes the builder has just
  made, so what it shows is the files' pixels from the first frame.

## Display

### Image Detail

The panel stops making a texture of the photograph and draws photograph tiles:

- **Which level.** With `s` the panel's physical screen pixels per level-0
  pixel (`effective_scale` times the pixels per point), the level is
  `round(log2(1 / s))` clamped to the photograph's levels, so a displayed
  texel covers between 0.71 and 1.41 screen pixels; at `s ≥ 0.71` it is level
  0, the original's own pixels. The coordinate transform, the overlays and
  the bench layer keep working in level-0 pixels; level `l`'s tile at column
  `c` and row `r` is drawn over level-0 pixels `c·512·2^l` to
  `(c+1)·512·2^l` (and the same for rows), clipped to `w_l·2^l`, so the
  dropped columns at the right and bottom edge are less than one texel wide.
- **Which tiles.** Those that intersect the panel's visible rectangle, asked
  for with `peek_tile`; a miss requests the tile on the rayon pool (it is
  milliseconds of work) and repaints when it lands.
- **While tiles load.** Each missing tile's area is drawn from the nearest
  coarser level that has its area loaded, ending at the coarsest resident
  level, so the panel is never blank once an image has been shown, and the
  image sharpens tile by tile.
- **Textures.** Each loaded tile becomes one egui texture of 514 x 514 with
  linear filtering, drawn over its core with texture coordinates inset by one
  texel, so neighbouring tiles blend across their edge as one texture would.
  The panel keeps at most 256 tile textures (about 270 MB of GPU memory),
  least recently drawn first out, and uploads at most 16 new ones per frame
  (about 17 MB), so a jump to a new place spreads its uploads over a few
  frames instead of stalling one.
- **Switching images** (`,` / `.`, the strip, animation). The new image's
  coarsest resident level is drawn at once if its mipmaps or pyramid have been
  read this session, then its tiles at the current view. While animation plays,
  the next image's tiles for the current view are asked for one frame ahead.
  The view persistence rule (§ "View persistence across image, reconstruction
  and panel changes") is unchanged; it computes `display_size` from the
  photograph's level-0 size, which the panel reads from the header (or the
  camera) rather than from a texture.
- **Zoom limit.** `MAX_ZOOM` is 32 times the fit. For a 16320 px photograph in
  a 400 px panel that reaches 0.78 screen pixels per level-0 pixel, so the
  original's pixels cannot be enlarged at all. The open questions propose a
  limit that also allows 8 screen pixels per level-0 pixel.
- **Agents.** A `screenshot` of the panel shows what the frame drew, coarser
  tiles included. `get_image_detail_view` gains `level`, the level the panel
  draws, and `tiles_pending`, how many visible tiles are not yet drawn at that
  level, so an agent waits for `0` before a screenshot it means to judge pixels
  by ([gui/mcp-server.md](../gui/mcp-server.md) § "The Image Detail view").

### The 3D view's camera-view background

The background becomes one texture of one level, made off the GUI thread:

- **Level.** The coarsest level whose longer side is still at least the
  viewport's longer side in physical pixels, but no finer than the first level
  whose longer side fits `max_texture_dimension_2d`. At 3840 px of viewport
  that is level 2 (4080 x 3060) of the 200-megapixel photograph; at 1920 px,
  level 3.
- **Source.** For an in-core photograph the level from its pyramid; for an
  out-of-core one the level's mipmap decoded whole, or level 0 from the
  original when that is what the rule picks. It is asked for through the
  cache on the rayon pool, as Image Detail's tiles are, and the coarsest level
  is drawn until it lands. A photograph that cannot be read is remembered as
  a failure, so it is not read again on every frame.
- **Texture coordinates.** The mesh's texture coordinates are scaled by
  `width / (w_l·2^l)` and `height / (h_l·2^l)`, so the dropped edge columns do
  not shift the picture.
- **Alpha.** The background reads RGBA today, but its pipeline draws with no
  blending (`blend: None` in `scene_renderer/pipelines/bg_distorted.rs`, and
  the shader returns the sampled colour), so the alpha has no visible effect.
  It reads the cache's RGB, with alpha 255, for every photograph, and nothing
  on screen changes; mipmaps having no alpha costs nothing here.

### GPU limits

The viewer keeps asking for `wgpu::Limits::default()`. Once no texture is
larger than a tile or the background's chosen level, nothing needs more, and
the default keeps the atlases' paging the same on every machine. Asking for
the adapter's own limits (16384 on most desktop GPUs) would raise the
background's finest level by one on some machines and change nothing else.

### Thumbnails

A display thumbnail is made from the photograph only when neither a `.sift`
nor the reconstruction carries one
([display_thumbnails.rs](../../crates/sfm-explorer/src/display_thumbnails.rs),
`display_thumbnail_row` in
[thumbnail.rs](../../crates/sfmtool-core/src/reconstruction/thumbnail.rs)). For
an out-of-core photograph it is made from the coarsest mipmap rather than from
a whole decode, which turns 0.8 to 1.7 s into a few milliseconds. Its pixels
differ slightly from a thumbnail of level 0; it is for display only.
Thumbnails written into files (`xform --add-thumbnails`, `.sift`) keep coming
from level 0.

## Every reader of photograph pixels

The readers as they are today (all paths under `crates/` unless written
otherwise), what each reads after this design, and whether its results change.
"Mipmaps" means levels 1 to `K` of an out-of-core photograph read from its
mipmaps; "depth" means the change of pyramid depth above.

| # | Reader | Today | After | Results change |
|---|--------|-------|-------|----------------|
| 1 | Image Detail (`dock.rs` → `display_photograph` → `load_image`) | level 0 as one texture | photograph tiles of the zoom's level; coarsest resident | display only; no texture over 514 px |
| 2 | 3D camera-view background (`bg_image.rs` `load_bg_texture`, from `app.rs`) | whole RGBA, one texture, decoded on the GUI thread | one level fitting viewport and limit, off the GUI thread, through the cache | display only; the alpha it reads today is not drawn |
| 3 | Track View's requests (`dock.rs` `cache_bench_track_images`, `cache_selected_point_images`) | a pyramid request per image per frame | the same, admitted within the budget | no |
| 4–6 | Track View's patch tiles, cluster-stage tiles, hover context (`track_view/body/mod.rs` `ensure_tile`, `ensure_context`; `tile.rs`, `patch.rs` → `render_tile`) | whole pyramid, `peek` | whole pyramid with mipmaps; view-region pyramid in the other draft | mipmaps for out-of-core; depth |
| 7–9 | Track View's crops, the Track Table's crop column and hover (`patch_pictures/crop.rs`, `track_table_panel.rs` `crop_picture`) | one level of the pyramid | the same level, from mipmaps for out-of-core | mipmaps; depth past a step of 64 |
| 10 | Track Table reference picks (`track_table_panel/picks.rs` → `render_reference` → `render_view_tile`, `clipped_share`) | `get`, whole pyramid; level 0 inside the outline | the same; level 0 is the original | mipmaps for the tile; `clipped_share` unchanged |
| 11–13 | Display patch bitmaps at open, conversion to embedded patches, `sfm web-export` (`display_bitmaps.rs`, `state/edits.rs` `convert_to_embedded_patches_job`, `web_export.rs`) | `get_many`, 6 levels; web-export with budget 0 | `get_many`, full depth, mipmaps; web-export makes missing mipmaps in the workspace | mipmaps; depth |
| 14–18 | The bench's steps and searches, live evaluation, *Find Nearby Tracks*, *Create Track Here*, *Add image to tracks*, *Refine all normals*, *Refit all points* (`bench.rs`, `state/edits.rs` `fetch`, `bench/nearby_tracks.rs`, `bench/track_at_pixel.rs`, `add_image_to_tracks.rs`, `state/edits/refine_all_normals.rs`, `refit_all_points.rs`) | `get_many`, levels per sample or per grid, `clipped_share` on level 0 | the same, mipmaps for out-of-core | mipmaps; depth |
| 16a | *Find Nearby Tracks*' grey reads (`GreyImages`, `bench/nearby/grey.rs`) | level 0 of each view, blurred grey `f32`, whole | unchanged (level 0 is the original); 800 MB per 200-megapixel view, which the other draft addresses | no |
| 19 | The viewer's cluster-patches build (`cluster_patches.rs` `read_photographs`) | `read_rgb`, full depth, outside the cache, all at once | `get_many` through the cache, so the CLI and the viewer read the same levels | mipmaps for out-of-core |
| 20 | Display thumbnails when none is stored (`display_thumbnails.rs` → `display_thumbnail_row`) | whole decode, resized to 128 | in core unchanged; out of core from the coarsest mipmap | display only |
| 21 | `.sift` building (`features/sift/files.rs` `build_sift_file`) | whole decode | whole decode, then the mipmaps from it | no |
| 22 | Image readings at save (`reconstruction/image_readings.rs`) | whole decode | unchanged | no |
| P1 | `sfmtool.fileio.read_image_rgb` / `read_image_rgba` (`sfmtool-py/src/fileio/image.rs`) | whole decode | unchanged: they read files, not pyramids | no |
| P2 | `ImagePyramidSet(recon, images)` and the kernels' `images` lists (`sfmtool-py/src/patches/views.rs`) | exact pyramids of the arrays passed | unchanged; a new `ImagePyramidSet.read(recon)` reads by path through the core cache | no for arrays; mipmaps through `read` |
| P3 | `embed-patches`, the strip montages, the `xform` steps that read photographs (`src/sfmtool/_embed_patches.py`, `src/sfmtool/strips/`, `src/sfmtool/xform/_images.py`) | `read_workspace_image` arrays into the kernels | `ImagePyramidSet.read(recon)` | mipmaps for out-of-core, so they match the viewer |
| P4 | `sfm cluster-patches` (`src/sfmtool/_cluster_patches.py`) | `read_image_rgb` arrays, full depth | through the same reader as #19 | mipmaps for out-of-core |
| P5 | The view-tile binding (`sfmtool-py/src/patches/view_tile.rs`) | an exact pyramid of the array passed | unchanged | no |
| P6 | Optical flow's pyramids (`sfmtool-py/src/flow/warp.rs`) | a grey pyramid of its own from an array | unchanged; flow reads level 0 | no |

Every reader that reads a pyramid of an out-of-core photograph now reads the
same levels whether it runs in the viewer or on the command line, and in any
session; and a reader of level 0 reads the original's pixels exactly as before.

## Standards

- **Progress.** `build_mipmaps` takes `&Progress`, counts levels and polls for
  cancellation between the decode, the reduction and each encode; a cancelled
  build writes nothing further and leaves no partial file. `PhotographCache::get`
  gains a `&Progress` parameter, since a `get` can now make a set (2 s or more)
  and already can take over a second to decode a 200-megapixel photograph;
  callers with nothing to report pass `&Progress::none()`. `get_many` passes its
  own down. `sfm ws mipmaps` reports through a `ProgressCounter`.
- **No panics on caller input.** A mipmap that is missing, cut short, damaged,
  or recording another photograph is stale and is made again; an original
  that cannot be read is an `Unreadable` answer, as today. A photograph whose
  level 1 would not fit a JPEG frame is in core by the tier rule, so
  `MipmapError::TooLarge` is only the encoder's own refusal, reported rather
  than unwrapped. The tile decoder's slicing is checked against the index,
  never trusted from the file.
- **Points at infinity.** Unaffected: the design changes where pixels come
  from, not any geometry, and every projection that picks a region already
  handles `w = 0` points.
- **Logic in Rust.** The naming rule, the record, the staleness test, the
  making and the reading are Rust in `sfmtool-core`; Python calls bindings.
- **Settings structs.** `MipmapOptions` and `PhotographCacheOptions`.
- **Test cost.** Every test runs on synthetic images of a few hundred pixels,
  with `MipmapOptions` setting a threshold and a tile size small enough that
  they have several levels and many tiles. No test reads or makes a large
  photograph.

## Interface

New, in `crates/sfmtool-core/src/camera/mipmap.rs` (with submodules for the
record and the tile decode), exported from `sfmtool_core::camera`:

```rust
pub const OUT_OF_CORE_LONG_SIDE: u32 = 4096;
pub const PHOTOGRAPH_TILE_SIZE: u32 = 512;
pub const MIPMAP_JPEG_QUALITY: u8 = 95;
pub const MIPMAP_RESTART_INTERVAL_MCUS: u16 = 16;
pub const CACHE_DIR_NAME: &str = ".sfm-cache";

/// The tier threshold and tile size. Production uses the default; tests shrink
/// both so a small synthetic image has several levels and many tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MipmapOptions {
    pub out_of_core_long_side: u32,
    pub tile_size: u32,
}

/// Every level's size for a photograph's size, and how many are stored.
pub struct PyramidLayout { /* … */ }
impl PyramidLayout {
    pub fn for_size(width: u32, height: u32, options: &MipmapOptions) -> Self;
    pub fn levels(&self) -> usize;                 // full_levels
    pub fn level_size(&self, level: usize) -> (u32, u32);
    pub fn stored_levels(&self) -> usize;          // K, 0 when in core
    pub fn is_out_of_core(&self) -> bool;
    pub fn tiles(&self, level: usize) -> (u32, u32);
}

/// Where a photograph's mipmaps are, or why it has none.
pub enum MipmapLocation {
    Files { root: PathBuf, relative: String, paths: Vec<PathBuf> },
    InMemory(NoFilesReason), // no workspace, `..`, both name forms taken
}
pub fn mipmap_location(photograph: &Path, layout: &PyramidLayout) -> MipmapLocation;

/// Make a photograph's mipmaps from one decode, writing what can be written.
/// `decoded` is `ImageU8::read_rgb(photograph)` when the caller already holds
/// it (`build_sift_file` does), and is decoded here otherwise. Returns the
/// encoded bytes of every stored level, which the caller decodes rather than
/// reading the files back.
pub fn build_mipmaps(
    photograph: &Path,
    decoded: Option<&ImageU8>,
    options: &MipmapOptions,
    progress: &Progress<'_>,
) -> Result<BuiltMipmaps, MipmapError>;

/// One mipmap's bytes, record and restart index, read and checked once.
pub struct MipmapFile { /* … */ }
impl MipmapFile {
    pub fn open(path: &Path, expected: &ExpectedMipmap) -> Result<Self, MipmapError>;
    pub fn from_bytes(bytes: Arc<[u8]>, expected: &ExpectedMipmap) -> Result<Self, MipmapError>;
    pub fn record(&self) -> &MipmapRecord;
    pub fn decode(&self) -> Result<ImageU8, MipmapError>;
    /// Pixels `x0..x1` by `y0..y1` of the level, equal to the same pixels of
    /// `decode`.
    pub fn decode_region(&self, x0: u32, y0: u32, x1: u32, y1: u32) -> Result<ImageU8, MipmapError>;
}

/// The restart-interval fast path for an original JPEG, or `None` when the
/// file does not qualify.
pub struct OriginalRegions { /* … */ }
impl OriginalRegions {
    pub fn open(path: &Path) -> Result<Option<Self>, image::ImageError>;
    /// Equal to the same pixels of `ImageU8::read_rgb(path)`.
    pub fn decode_region(&self, x0: u32, y0: u32, x1: u32, y1: u32) -> Result<ImageU8, image::ImageError>;
}

pub enum MipmapError { Io { path: PathBuf, error: std::io::Error }, Stale { path: PathBuf, why: String },
    Source(image::ImageError), TooLarge { width: u32, height: u32 }, Encode(String), Cancelled }
```

Changed, in [photograph_cache.rs](../../crates/sfmtool-core/src/camera/photograph_cache.rs):

```rust
pub struct PhotographCacheOptions {
    pub budget_bytes: u64,          // default_budget_bytes()
    pub tile_budget_bytes: u64,     // default_tile_budget_bytes()
    pub mipmaps: MipmapOptions,
}

impl PhotographCache {
    pub fn new(options: PhotographCacheOptions) -> Self; // no `levels`: depth is by size
    pub fn get(&self, path: &Path, progress: &Progress<'_>) -> Option<Arc<ImageU8Pyramid>>;
    // peek, peek_state, get_many, insert, forget, clear, stats as today

    /// The photograph's level sizes, once its header has been read.
    pub fn layout(&self, path: &Path) -> PeekedLayout;
    pub fn peek_tile(&self, path: &Path, key: TileKey) -> PeekedTile;
    pub fn get_tile(&self, path: &Path, key: TileKey, progress: &Progress<'_>) -> Option<Arc<PhotographTile>>;
    /// One level whole, for the 3D background: from the pyramid entry in core,
    /// the mipmap or the original out of core.
    pub fn get_level(&self, path: &Path, level: usize, progress: &Progress<'_>) -> Option<Arc<ImageU8>>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TileKey { pub level: u8, pub col: u32, pub row: u32 }

pub struct PhotographTile {
    pub key: TileKey,
    /// The core's top-left pixel and size in level pixels.
    pub origin: [u32; 2],
    pub size: [u32; 2],
    /// RGB, `size + 2` on each axis: the core with a one-pixel border.
    pub pixels: ImageU8,
}

pub enum PeekedTile { Ready(Arc<PhotographTile>), Unreadable, NotReady }
pub enum PeekedLayout { Known(Arc<PyramidLayout>), Unreadable, NotKnown }
```

`DISPLAY_PYRAMID_LEVELS` and the viewer's `state::PYRAMID_LEVELS` are removed.
`render_display_patch_bitmaps` documents that it reads pyramids of the cache's
depth.

The viewer's side, in
[photograph_requests.rs](../../crates/sfm-explorer/src/state/photograph_requests.rs):
`display_photograph` keeps its contract; a `display_tile` beside it peeks a
tile and starts one `get_tile` on a miss; and the admission rule for pyramid
requests lives there. The level builder is `state/level_builder.rs`.

## Python bindings

- `sfmtool.fileio` gains nothing: mipmaps are not a format a caller reads.
- `ImagePyramidSet.read(views_or_recon, workspace_dir=None, progress=None)`:
  the set of pyramids of the reconstruction's photographs, read by path
  through a `PhotographCache` of budget 0, so each is read the way the viewer
  reads it, making missing mipmaps on the way. The `images` argument of the
  constructor and of every kernel keeps meaning exact pyramids of the arrays
  given.
- `sfmtool.workspace.build_mipmaps(paths, progress=None)`: makes the mipmaps of
  the out-of-core photographs among `paths`, the binding behind
  `sfm ws mipmaps` and the Python SIFT extractors' call.

## Implementation steps

Each is one pull request, useful on its own, with its tests. Steps 1 to 3 end
the crash with in-memory pyramids, before any file is written.

1. **Depth by size.** Pyramids of `full_levels` depth everywhere;
   `PhotographCache::new` loses `levels`; `DISPLAY_PYRAMID_LEVELS` goes.
   Tests: the depth for several sizes; a kernel render that clamped at level 5
   now reads level 6, on a synthetic view shrunk 64 times; the tests that pin
   rendered values updated where they change, each change explained.
2. **Image Detail draws photograph tiles** cut from the in-memory pyramid,
   with the level rule, the coarser fallback, the per-frame upload cap and the
   texture limit of 256. This alone ends the crash in Image Detail for every
   photograph, though a 200-megapixel one is still read whole until step 6.
   Tests (headless, the panel's lib tests): the level for a set of
   scales; the tiles a view asks for; a 9000 x 16 synthetic photograph never makes
   a texture over 514 px; the fallback draws the coarser level where a tile is
   missing; the test data's photographs look as they did.
3. **The 3D background** from one level by the viewport and the limit, off the
   GUI thread, through the cache, with a remembered failure. Ends the crash in
   camera view. Tests: the level rule; the texture coordinates' scale for odd
   sizes; an unreadable photograph is read once.
4. **Mipmap files.** Image discovery skips `.sfm-cache` (`expand_paths`, the
   camrig and rig-config finders) with a test each; then `PyramidLayout`,
   `mipmap_location`, the record, `build_mipmaps`, `MipmapFile::open` and
   `decode`, the staleness test, the writers' rules, `.gitignore` and
   `CACHEDIR.TAG`. Tests on 300 x 200 synthetic photographs with a threshold of
   64 and a tile of 32, in a temporary workspace: the name for `a.b.png`, a
   name with no extension, `x.jpg` beside `x.png` (second form), both forms
   taken (in memory), a `..` name and no workspace (in memory); two builds give
   the same bytes; each staleness case (length, time with equal hash, time with
   another hash, another `source`, cut short, no record) and that a stale set
   is made again; a read-only cache directory gives the in-memory equivalent
   with the same pixels; two threads and two processes building at once; the
   peak memory of one build, measured.
5. **Tile decode of a mipmap**: the restart index and `decode_region`. Tests:
   every tile of several synthetic levels with odd sizes and a restart
   interval of 2 and 16 MCUs, including the short last interval, equals the
   same pixels of `decode`; a grey mipmap; a damaged interval is `Stale`.
6. **The out-of-core tier in the cache**: pyramid entries from mipmaps,
   photograph tiles from mipmaps and level 0, the tile budget and its two
   eviction exceptions, `get` with `&Progress`, `get_level`, and the request
   admission rule in the viewer. Image Detail and the background read
   out-of-core photographs through it. Tests: with the small threshold, a
   `get` on a photograph with no mipmaps writes them and returns their pixels;
   a second cache reads them without decoding the original's levels; tiles
   equal the pyramid's levels; the tile budget evicts tiles before the coarsest
   residents; admission refuses what does not fit, with a fake budget.
7. **Makers**: `sfm sift` and `build_sift_file`, the viewer's level builder
   (priority, memory-sized concurrency, the panel line), `sfm ws mipmaps` with
   `--prune` and `--check`, and the thumbnail from the coarsest mipmap. Tests:
   `sfm sift` on a small-threshold synthetic workspace writes the files;
   the builder's order follows the selection; `--prune` deletes an orphan and
   keeps a current set; `--check` writes nothing.
8. **Readers that must agree with the viewer**: the viewer's cluster-patches
   build through the cache, `ImagePyramidSet.read`, and `embed-patches`, the
   strips, the `xform` steps and `sfm cluster-patches` on it. Tests: with the
   small threshold, the CLI and the viewer's code give the same pyramids for
   the same photograph; in-core results unchanged.
9. **The level-0 fast path**: `OriginalRegions`. Tests: JPEGs written for the
   test by `jpeg-encoder` with restart intervals at 4:2:0, 4:4:4 and grey, of
   odd sizes, every region (including edges and corners) equals the same
   pixels of `read_rgb`; a progressive file, one with no restart interval, and
   one whose interval does not divide the row are refused.

Then [view-region-pyramid.md](view-region-pyramid.md), if the memory of whole
pyramids for the kernels matters on the machines in use.

## Measurements

The scratch program behind the figures, kept out of the tree: a binary crate
depending on `jpeg-encoder = "=0.7.1"` (feature `simd`) and
`jpeg-decoder = "=0.3.2"` (no default features), built `--release`. It decodes
the photograph, makes the levels with a copy of `downsample_2x`, encodes level 1
at quality 90 and 95, 4:4:4 and 4:2:0, with restart intervals of 0, 8, 16 and
64 MCUs, decodes each back (time, mean absolute error against the exact level),
encodes every level at both qualities, indexes the restart markers of the
original and of level 1, and decodes 512 px regions as above, comparing each
with the whole decode pixel by pixel. Times are wall clock of one run each,
repeated two to four times; the ranges in this draft are over those runs.

## Open questions

1. **The name of a mipmap.** The decided rule (`<base>-mipmap2.jpg`) needs the
   collision handling above for `x.jpg` beside `x.png`. Options: (a) keep it,
   with the second form on collision, as designed; (b) always use the whole
   file name, `IMG_0001.JPG-mipmap2.jpg`, which cannot collide and matches how
   a `.sift` is named (`IMG_0001.JPG.sift`). Recommendation: (b), because a
   name that is one-to-one removes a rule, a fallback and an order dependence,
   at the cost of a doubled extension in the name.
2. **The threshold.** Options: a long side over 4096 (as designed), over 8192
   (only photographs no texture can hold), a pixel count (say 16 MP), or every
   photograph. Recommendation: 4096, for the reasons above; every photograph
   would write files for the test data and make every small capture pay for an
   encode with nothing to gain.
3. **Quality.** 95 (as designed: 15.2 MB per 200-megapixel photograph, 0.49 grey
   levels of error at level 1) or 90 (10.2 MB, 0.73). Recommendation: 95.
4. **Should the Python SIFT path hand its decode to the mipmap binding?** It
   saves one decode per out-of-core photograph (0.8 to 1.7 s) but the binding
   must then trust that the array is the decode of that file. Recommendation:
   decode again; the extraction dominates, and a binding that reads by path
   cannot be handed the wrong pixels.
5. **The command's name.** `sfm ws mipmaps` (as designed), a top-level
   `sfm mipmaps`, or no command and only `sfm sift` and the viewer.
   Recommendation: `sfm ws mipmaps`, since the files belong to the workspace.
6. **The level builder in the Background Tasks panel.** A line of its own (as
   designed), a quiet task, or nothing. Recommendation: the line; a quiet task
   would wait behind every other quiet task and give way to every asked-for
   one, and the display waits on this work.
7. **Image Detail's zoom limit.** `MAX_ZOOM` = 32 times the fit does not reach
   the original's pixels of a 200-megapixel photograph in a small panel.
   Options: keep it; make the limit `max(32, 8 / fit_scale)` so a level-0 pixel
   can always be enlarged to 8 screen pixels. Recommendation: the second.
8. **A tiled lossless copy of level 0** for originals that the fast path does
   not cover (no restart interval, PNG, TIFF), so zooming to full resolution
   does not cost a whole decode. It contradicts decision 1. Recommendation: no;
   keep the whole decode as one cache entry, and revisit only if a capture of
   such originals is in use.
9. **Mipmaps written by another build.** A file is the record, so a newer
    `jpeg-encoder` never rewrites it; but a read-only copy of the workspace
    read through the in-memory equivalent by another build can differ from the
    files by the encoders' differences. Options: accept it (as designed), or
    rebuild when the record's `encoder` differs. Recommendation: accept it and
    keep `encoder` in the record for diagnosis; rebuilding would change results
    with no input changed.
10. **The decoder.** The tile decode needs `jpeg-decoder`'s
    `ColorTransform::None`, and the fast path its present upsampling and
    conversion. [reading-and-writing-images.md](../core/camera/reading-and-writing-images.md)
    plans to move JPEG reading to zune-jpeg when it is fixed. Recommendation:
    the mipmap reader keeps `jpeg-decoder` until zune-jpeg passes the same
    tile-equality tests; the fast path follows whichever decoder `read_rgb`
    uses, and turns off if its test fails.
11. **Budgets.** The tile budget's default (a sixteenth of memory, 512 MiB to
    4 GiB) and the 256 GPU tile textures are estimates from the figures above.
    Recommendation: ship them and report the cache's `stats`, with the tile
    entries, in the Background Tasks panel's tooltip so they can be judged on
    real captures.

## Problems found with the decisions

- **The decided name is not one-to-one** (open question 1). Designed around
  with a record in every file and a second name form.
- **Mirroring the workspace does not cover every photograph.** A photograph
  with no workspace above it, or with `..` in its image name, has no mirror
  path. The in-memory equivalent gives those the same pixels without files.
- **"Every later reader reads the files" needs a rule for when files cannot be
  written.** The in-memory equivalent is that rule; its pixels equal the files'
  only with the same build (open question 9).
- **A coarsest level of about one tile is not deep enough for the kernels.**
  They need levels to about one pixel. The levels below the coarsest mipmap are
  computed in memory from it, which keeps every reader consistent because the
  input is the file.
- **Level 0 as the original makes full-resolution regions as fast as the
  original allows.** For the user's photographs (restart interval 4 MCUs) a
  region decodes in about 3 ms; for an original with no restart interval, or
  a PNG or TIFF, the first look at full resolution costs a whole decode, 0.8 s
  or more and 571 MiB at this size (open question 8).
- **The mipmap directory is reachable by image discovery.** A directory scan
  at the workspace root would read mipmaps as images; step 4 makes every
  finder skip it before anything writes there.
