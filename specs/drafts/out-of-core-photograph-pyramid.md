# Out-of-core photograph pyramid

**Status:** Draft

Decided by the user, and followed here: full resolution is never stored again
for an original that can be read by region (a JPEG whose restart markers allow
it, § "Which originals can be read by region"), whose level 0 is the original
file itself; for every other out-of-core photograph, full resolution is stored
once as a tiled re-encode, `<file name>-mip0-q<quality>.jpg`, and every pyramid reader (the
display and every kernel) reads level 0 from that file instead of the
original, while the readers that decode a whole original and read no pyramid
(SIFT extraction, the image readings at save, `read_image_rgb` and the
commands built on it) keep reading the original (§ "What still decodes an
original whole"); the reduced levels of
a large photograph are JPEG files in `.sfm-mipmaps/` at the workspace root, one
file per level, in directories that mirror the workspace; once a photograph has
these files every reader, the display and the patch kernels alike, reads its
reduced levels from them; the unit of display and of region reads is a 512 px
square; there is no stopgap fix; and a kernel reading an out-of-core
photograph reads only the region of each level that one render of one patch
needs, so its memory grows with the patches it renders and not with the size
or number of the photographs. That last part is specified in the companion
draft [view-region-pyramid.md](view-region-pyramid.md), which this design
requires: without it no kernel can read a large photograph within a bounded
memory, so it is a numbered step here (§ "Implementation steps"), not a later
optimization. The user has decided every open question of this draft
(§ "Open questions", 1 to 18) and every one of the companion draft's.

Amends [core/camera/photograph-cache.md](../core/camera/photograph-cache.md)
(the cache hands out photograph tiles for every photograph instead of whole
pyramids, and gains an on-disk tier),
[core/camera/reading-and-writing-images.md](../core/camera/reading-and-writing-images.md)
(a second JPEG writer and reader, for mip level files),
[gui/multi-panel-image-browser.md](../gui/multi-panel-image-browser.md)
§ "Image Detail" (the panel reads its photograph tiles from the cache),
[gui/mcp-server.md](../gui/mcp-server.md) § "The Image Detail view" (two
fields), and
[workspace/workspace.md](../workspace/workspace.md) (the `mipmap_options`
object, and the mipmap directory in § "Workspace Directory Layout"), and
[gui/background-tasks.md](../gui/background-tasks.md) (the quiet task *Build
mipmaps*: a row in its operations table and a fifth quiet operation in
§ "Quiet tasks"), and
[gui/camera-views.md](../gui/camera-views.md) § "Photograph background" (the
level assembled from photograph tiles). Each links back here.

SfM Explorer shows a photograph, and the patch kernels sample it, through one
image pyramid per photograph: the photograph and successively halved copies of
it. Today each pyramid is built whole in memory from one decode of the file,
and Image Detail uploads level 0 as a single GPU texture. That works for the
test data and for 4K video frames, but a 200-megapixel phone photograph
(16320 x 12240) is 571 MiB as RGB and 762 MiB as a pyramid, and wider than the
8192 px texture the viewer's GPU device allows, so selecting one crashes the
viewer. This draft makes the size of a photograph matter only in how long its
first read takes: the display draws 512 px tiles of the pyramid level that
matches its zoom, so it never makes a texture larger than a tile and shows
full-resolution pixels when zoomed in far enough; and every level of a large
photograph, level 0 included, can be read a tile at a time, either from the
original or from JPEG files computed once and stored beside the workspace, so
that a session that looks at a photograph decodes only what it shows, and every
later session and every kernel reads the same levels.
Every photograph, whatever its size, is read through photograph tiles: no
reader holds a whole pyramid, each render reads a view-region pyramid
assembled from tiles, and a kernel that renders many patches keeps only the
regions its renders need (§ "The caches in memory"). An in-core photograph,
one without mip level files, is decoded whole when a tile of it is needed and
cut into tiles; only the tiles stay.

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
| a pyramid of 6 levels (the viewer's depth before step 1) | 798.8 MB (761.8 MiB) |

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

These are proposed for [GLOSSARY.md](../GLOSSARY.md), each with its reason;
*photograph tile* is in it since step 2, and *mipmap*, *mip level*, *mip level
file*, *mipmap directory*, *region-readable* and *out-of-core* / *in-core
photograph* since step 5, with shorter reasons there; the rows below keep the
fuller argument and the names later steps add (`sfm ws mipmaps`, the *Build
mipmaps* task, `MipmapLocation`).

| Word | Means | Not | Why |
|------|-------|-----|-----|
| **mipmap** / **mip level** / **mip level file** | a photograph's **mipmap** is its whole stored chain of levels, kept under `.sfm-mipmaps/`: its reduced levels, and its full-resolution copy when the original cannot be read by region. One stored level is a **mip level**, numbered as the pyramid's level is (level 0 full resolution, level `l` reduced by `2^l`), and the file holding it is a **mip level file**, `<file name>-mip<l>-q<quality>.jpg`: `IMG_0001.JPG-mip1-q95.jpg` is half size at quality 95, `-mip2-q95.jpg` a quarter, and `-mip0-q95.jpg` the full-resolution copy; this draft names a file by its level part alone (`-mip0`, `-mip1`) where the quality does not matter. The number after `mip` is the level index the code uses; no factor is converted to a level anywhere. `sfmtool_core::camera::mipmap` | `mipmap set`, a single file called a mipmap, `level file`, `cache file`, `pyramid file`, factor names such as `-mipmap2` | the user chose the standard usage, so others find it familiar: Williams, "Pyramidal Parametrics" (SIGGRAPH 1983), coined *mip map* for the whole prefiltered set of an image at successive halvings; OpenGL says a texture has a mipmap whose images are *mipmap levels*, numbered by level of detail with level 0 the base image (`glTexImage2D`'s `level`); Direct3D and Vulkan say *mip chain* and *mip level* (`mipLevels`); KTX and DDS store a whole chain in one file. *Level* alone stays the in-memory pyramid level, which for a small photograph has no file. Names about the whole feature keep *mipmap* (`mipmap_options`, `sfm ws mipmaps`, `build_mipmaps`, `MipmapError`, `MipmapLocation`, `sfmtool_core::camera::mipmap`), since they act on photographs' mipmaps, plural in the standard sense. The viewer, the CLI and the Python messages use the same words as the code, with no separate user-facing ones: the quiet task *Build mipmaps*, the line *Building mipmaps: {done} of {total}*, a panel's *Building mipmap…* while the one photograph's mipmap it waits for is made, the phase *build mipmaps*, and *mip level* where a message is about one level. A mipmap is made inside the photograph cache's `get` or `get_many`, in the path's slot, with no builder object |
| **region-readable** (an original) | an original JPEG whose layout passes the test in § "Which originals can be read by region", so a rectangle of it decodes to exactly the pixels of the whole decode; its level 0 is the original, and it has no `-mip0` | `tiled`, `seekable`, `RST-capable` | names what the test decides, that a region can be read, rather than one ingredient of it |
| **mipmap directory** | `.sfm-mipmaps/` at a workspace's root, holding the mip level files of the workspace's out-of-core photographs, in directories that mirror the workspace; it can be deleted at any time and rebuilt | `cache directory`, `.sfm-cache`, `cache folder`, `.cache` | the name says what is in it, mipmaps, and `sfm` keeps it apart from other tools' directories at a workspace root |
| **out-of-core photograph** / **in-core photograph** | a photograph of a workspace whose file has `mipmap_options`, with a width or height over that object's `out_of_core_long_side` (1536 as `sfm ws init` writes it), is out of core (§ "Which photographs are out of core"; one with a side of 65536 px or more is refused everywhere): its reduced levels come from mipmaps, its level 0 from the original or a `-mip0`; any other photograph is in core: it has no mip level files, and its tiles are cut from a whole decode of the original, made when a tile of it is needed and not kept. Every photograph, in core or out, is read a region at a time through photograph tiles | `large image`, `tiled image`, `big photograph` | the tier decides where the levels live, on disk or only in memory, which is what *core* means here; *large* says nothing about what is done differently, and every photograph is read in tiles |
| **photograph tile** | a 512 x 512 block of one level of a photograph's pyramid, with a one-pixel border of its neighbours' pixels to assist bilinear interpolation: the unit the display draws as one texture and that a level is read in; `PhotographTile`, keyed by level, column and row | bare `tile`, `block`, `chunk` | bare *tile* is a view's tile, the patch a kernel renders ([GLOSSARY.md](../GLOSSARY.md) § "Rendering a view's tile"), and Track View draws those beside photographs, so the photograph's blocks need a qualified name |

## Which photographs are out of core

The rule, and the refusal of a photograph with a side of 65536 px or more,
are built (step 5) and specified in
[core/camera/mipmaps.md](../core/camera/mipmaps.md#which-photographs-are-out-of-core):
the refusal is `GetError::TooLarge { path, width, height }` from
`PhotographCache::get`, decided from the header before any decode in every
workspace. Still to come with the readers that raise it:
`PhotographReadError::TooLarge` for a kernel (step 10) and the Python
bindings raising it as `OSError`, as `read_image_rgb` raises a decoder's
limit (step 16). Under the default threshold of 1536 px, the test data's
photographs are in core except `dino_dog_toy`'s 2040 x 1536, which are out of
core in a new workspace (the checked-in workspaces have no object, so every
photograph is in core there); 1920 x 1080 video frames are out of core, as are
3840 x 2160 frames, 4032 x 3024 phone photographs, a 24-megapixel camera's
6000 x 4000 and the 200-megapixel phone photographs.

Why 1536 (open question 26): once every photograph is drawn in tiles (below),
the texture limit, 8192, no longer separates anything, and the tier is about
the time of a read and the memory of a job. An in-core photograph whose tiles
were evicted is decoded whole again, all its pixels at once, about 0.05 s at
3840 x 2160 but 0.2 to 0.4 s for a 6000 x 4000 photograph (estimated by pixel
count), while an out-of-core one decodes only the tiles it needs, about 3 ms
each, at the cost of one extra encode of its levels, once. At 1536 a job
reading many photographs of a few megapixels, such as those of a 24-megapixel
camera or a phone, reads them a tile at a time, so its memory is bounded by
the tiles it reads rather than by whole decodes.
The threshold is separate from `double_image_below_long_side`, which
`sfm ws init` sets to 4096 (an image is doubled for SIFT only when both sides
are below it), so a new workspace's photograph whose longer side is 1537 to
4095 px is out of core and doubled for SIFT. Nothing depends on the two
agreeing: SIFT decodes the original whole in either tier, and the mipmap is
made from the photograph, not the doubled image (by `build_sift_file` from its
decode before the extraction, and by `sfm sift` from the original once the
files are written).
The user has decided that the threshold is a workspace setting: which tier a
photograph is in decides which pixels the kernels read, so a workspace keeps
its own value if the default changes later, as it keeps its feature options,
and a change to it is a change of results that the workspace's owner makes
knowingly. Tests set a small threshold the same way, in the workspace file of
a temporary workspace ([mipmaps.md § "The workspace setting"](../core/camera/mipmaps.md#the-workspace-setting)).

## Levels by size

Two counts are decided by the photograph's size, not fixed.

**Pyramid depth.** Built (step 1): every photograph pyramid has
`ImageU8Pyramid::full_levels(width, height)` levels, 14 for 16320 x 12240,
as [core/camera/photograph-cache.md](../core/camera/photograph-cache.md)
describes. The levels the mipmap stores and the photograph tiles cut are those
levels.

**Mip levels stored.** Built (step 5): levels 1 to `K`, and `-mip0` exactly
when the original is not region-readable, as
[core/camera/mipmaps.md](../core/camera/mipmaps.md#levels-by-size) specifies,
with its table of `K` by size. The levels below `K` are to be box-filtered in
memory from the decoded level `K` when the cache reads mipmaps (step 14).

## Which originals can be read by region

Level 0 of an out-of-core photograph is read a tile at a time, by every
reader. An original that allows it is read directly; every other one gets a
a `-mip0` file, a tiled re-encode of full resolution, made once, and from then
on **every reader reads level 0 from `-mip0`, never from the original**:
the display, every kernel, `clipped_share`, the grey reads of *Find Nearby
Tracks*. This is the user's decision. No reader of level 0 holds a whole
decode of an original while it reads regions of it: every reader of level 0
holds only tiles.

**The test.** Built (step 5): `region_layout(bytes) -> Result<RegionLayout,
NotRegionReadable>`, with its seven rules, is specified in
[core/camera/mipmaps.md](../core/camera/mipmaps.md#which-originals-can-be-read-by-region).
Measured once (step 5), read-only: all 17 of the user's 16320 x 12240
photographs are region-readable, each 4:2:0 with `R = 4` of 1020 MCUs per row
and 195,075 intervals, and the test took 9 to 12 ms a photograph once its
bytes were in memory. A camera JPEG without a `DRI` segment, a progressive
JPEG and every PNG or TIFF fail and get a `-mip0`.

The test is part of the definition of each photograph's pixels: changing it
moves photographs between reading the original and reading `-mip0`, which
changes results, so it changes only with a test showing why. What a decoder
change that broke the region decode's equality would require is in
[core/camera/mipmaps.md](../core/camera/mipmaps.md#a-region-of-a-region-readable-original).

**What still decodes an original whole.** Making a photograph's mipmap
(once per mipmap, below); the readers that are not pyramid readers and are
unchanged by this design: SIFT extraction (`build_sift_file`, the Python
extractors, COLMAP), the image readings at save, `read_image_rgb` and
`read_image_rgba` and every command that reads whole arrays through them
(`undistort`, `to-nerfstudio`, optical flow's `load_gray`, the drawing
commands, `xform --add-thumbnails`); and the display thumbnail of an in-core
photograph. No pyramid reader of an out-of-core photograph does. These
readers keep the original because they run before the mip level files exist,
or store what they compute from the exact pixels, and reading a lossy copy
would change the features and readings every reconstruction is built from.
SIFT therefore reads the original's exact pixels while the kernels read a
non-region-readable photograph's level 0 from `-mip0`, which differs by
0.07 to 0.09 grey levels on average for a JPEG original (measured below); the
keypoints' positions are not affected by a difference that size in any way
this design relies on.

## Mip level files

Built (step 5) and specified in
[core/camera/mipmaps.md](../core/camera/mipmaps.md): where the files are and
their names, what is written in one and its record, the `mipmap_options`
object and `read_mipmap_options`, how one build makes a mipmap, the in-memory
equivalent, when a file is current, writers that meet, `CACHEDIR.TAG` and the
discovery skips, and why the files are not a file format. What the later
steps add to them:

- **The cache reads them** (step 14): it reads `mipmap_options` once per
  workspace root and again when the file's modification time changes; it
  checks the coarsest mip level file when it first needs a photograph and each
  finer one when it first opens it, so a session's start costs one small file
  per out-of-core photograph and one `stat` of each original; it takes the
  region answer from the coarsest file's record while test 4 holds, and an
  original read by region is indexed when its first level-0 tile is asked
  for, a failure of that index making the mipmap stale; it remembers a hash
  comparison for as long as it lives, for each original and spelling of its
  path (one `MipmapSource` kept per spelling); a photograph that the hash
  finds is not a shared file's source makes its own mipmap over it, and the
  reader logs a warning naming both paths; a build whose files could not be
  written is held by that cache as the in-memory equivalent, its bytes
  counted against the budget, with a warning in the log saying why.
- **An invalid object** (step 14): a photograph whose workspace file has an
  invalid `mipmap_options` is not read until the file is fixed; its readers
  report the file's error, `GetError::Workspace` from the cache (built,
  14a), and Image Detail says *Workspace file error:* and the message
  (14e). `load_workspace_config` calls `read_mipmap_options` through a
  binding, raising `RuntimeError` (step 16).
- **Writing the object** (steps 15 and 17): `sfm ws init` writes it with 1536,
  95 and 95; `sfm ws mipmaps --enable` adds it to an existing workspace (or
  with `--out-of-core-long-side`, `--jpeg-quality` and
  `--full-resolution-jpeg-quality`) and refuses a file that has it;
  `sfm ws upgrade` leaves it as it finds it; and the viewer says once per
  session, the first time it reads a photograph over 1536 px in a workspace
  without the object: *This workspace decodes large photographs whole. Run
  `sfm ws mipmaps --enable` to read them a tile at a time.*
- **Orphans** (built, step 15e, as
  [ws-mipmaps-command.md § Orphans](../cli/workspace/ws-mipmaps-command.md#orphans)
  says, with the cases step 15e's note adds): `sfm ws mipmaps --prune` deletes the mip level files
  whose record names a photograph that no longer exists or that they are
  stale for, those whose quality in the name differs from the workspace's
  current setting for their level, and temporary files older than an hour;
  in a workspace without the object it deletes every mip level file there.

### Measurements behind the files

At level 1 of the measured photograph the mean absolute error is 0.49 grey
levels at quality 95 and 0.73 at 90, for 10.8 MB against 7.3 MB; 4:2:0 would
save 14% (9.2 MB against 10.8 MB), and restart markers every 16 MCUs cost 2.4%
(10.77 MB against 10.52 MB). All five reduced mip level files of the measured
photograph at quality 95 are 15.2 MB together (10.77, 3.21, 0.90, 0.25,
0.07 MB), half the original's 30.0 MB; at quality 90 they are 10.2 MB.

**Level 0, `-mip0`.** Measured on two of the user's photographs, encoding
the whole decode of the original (16320 x 12240) with the settings above at
several qualities, and comparing the decoded result with that whole decode
(mean absolute error over every value, and the share of values off by 3 or
more):

| Quality | `164358.jpg` (30.0 MB) | `164648.jpg` (39.8 MB) | `164358.jpg` shifted by one pixel |
|---------|------------------------|------------------------|-----------------------------------|
| 90 | 24.2 MB, 0.42, 1.4% | 31.9 MB, 0.61, 2.9% | 22.7 MB, 0.47, 1.7% |
| 95 | 33.0 MB, 0.067, 0.06% | 43.4 MB, 0.094, 0.07% | 33.8 MB, 0.32, 0.66% |
| 97 | 40.0 MB, 0.088, 0.08% | 53.0 MB, 0.13, 0.11% | 42.9 MB, 0.23, 0.30% |
| 98 | 42.3 MB, 0.074, 0.04% | 56.1 MB, 0.099, 0.05% | 47.7 MB, 0.19, 0.17% |
| 100 | 61.6 MB, 0.053, 0.01% | 81.7 MB, 0.067, 0.01% | 72.9 MB, 0.085, 0.02% |

Encoding took 1.4 to 2.1 s and decoding 0.5 to 1.1 s at these qualities. The
first two columns are the common case for `-mip0`, a JPEG original without
usable restart markers: the re-encode uses the same 8 x 8 block grid as the
original, whose pixels are already the result of a quantised DCT on that grid,
so quality 95 re-encodes it to within 0.07 to 0.09 grey levels, closer than 97
does (the error is not monotonic in quality, since 95's tables fall near the
original's). The third column crops the decode by one pixel, so the blocks no
longer line up; it stands in for a source that has never been a JPEG (PNG,
TIFF), where `-mip0` is the first loss. There 98 halves the error of 95
(0.19 against 0.32) for 41% more bytes. The user has decided that
`sfm ws init` writes 95 for both keys, the best quality measured for JPEG
originals, and that no quality depends on the original's format (open
question 3); a workspace of PNG or TIFF photographs that wants the lower
error edits its `full_resolution_jpeg_quality` to 98 itself. The reduced
levels keep their own key because their
trade-off differs: they are box-filtered, never on the original's grid, and
small.

Disk per 200-megapixel photograph, at the defaults:

| Original | Files | Size, measured photograph |
|----------|-------|---------------------------|
| region-readable JPEG (the user's) | `-mip1` to `-mip5` | 15.2 MB, half the original |
| JPEG without usable restart markers | `-mip0` to `-mip5` | 33 to 43 MB + 15 MB = 48 to 58 MB, about 1.5 times the original |
| PNG or TIFF | `-mip0` to `-mip5` | about 34 MB + 15 MB = 49 MB at 95, 63 MB at 98 (estimated from the shifted column) |

For the user's 17 photographs that is about 260 MB (15.2 MB measured for one,
scaled by the others' sizes as an estimate); if they lacked restart markers it
would be about 0.9 GB. A capture of 500 photographs of 6000 x 4000 without
restart markers would be about 3 GB (estimated by scaling the 200-megapixel
figures by pixel count).

**The cost of a build.** Measured once (step 5) on `20261115_164358.jpg`, a
region-readable original, at quality 95 with a release build: the peak
working set of the process making its mipmap was 919 and 922 MiB in two runs,
against the estimate of about 1 GB from the buffer sizes (571 MiB for the
decode, the decoder's own component planes while it decodes, 190 MiB of
reduced levels, the 30 MB original held for its hash and region test, and
15 MB of encoded files). The whole build, read, decode, reduction, region
test and the encodes and writes of levels 1 to 5, took 1.77 and 1.78 s, and
wrote 10.77, 3.21, 0.90, 0.25 and 0.07 MB. A photograph that needs a `-mip0`
adds its encode, 1.4 to 2.1 s, and 33 to 43 MB of encoded bytes held until
written (measured separately, above).

## Reading pixels

### A tile of a mip level file

A photograph tile of level `l ≥ 1` is decoded from the level's mip level file
without decoding the rest of it. Built (step 6): the restart index and
`MipLevelFile::decode_region`, specified with the two choices the identity
needs (converting colour outside the decoder, and 4:4:4) in
[core/camera/mipmaps.md § "A tile of a mip level file"](../core/camera/mipmaps.md#a-tile-of-a-mip-level-file).

Measured before step 6 on level 1 of the photograph (8160 x 6120, quality 90,
interval 16 MCUs), nine tiles including the right and bottom edges: 2.3 to 6 ms
per tile on an unloaded machine (up to 15 ms under load), each identical to the
same pixels of a decode of the whole file through the same raw path; indexing
its 48,769 intervals took 2 to 7 ms. Measured again with step 6's code
(§ "Measurements"): nine 514 px tiles (512 px with
their one-pixel border) of level 1 built from one of the user's photographs,
among them all four corners and the right and bottom edges, took 2.9 to 4.3 ms
each at quality 90 and 3.1 to 4.7 ms at quality 95 (10.8 MB), each identical
to the same pixels of `decode`; opening and indexing the file took 2.2 to
3.7 ms. A whole `decode` of that level took 0.58 to 0.64 s, slower than the
decoder's own SIMD conversion.

**A whole level** is decoded from the whole file through the same raw path and
conversion. Only the coarsest mip level, level `K`, is decoded whole in use:
the levels below it are computed from that decode (§ "Levels by size"). A whole
decode of level 1 of this photograph to RGB took 0.16 to 0.45 s; the raw path
does the same work less the colour conversion, which it then does itself.

### A region of level 0

Every level-0 region of an out-of-core photograph is a tile decode, from one of
two files:

- **A region-readable original** (§ "Which originals can be read by region")
  is read directly, by `OriginalRegions` (built in step 6), whose margins for
  chroma upsampling and for the decoder's scalar line ends are specified in
  [core/camera/mipmaps.md § "A region of a region-readable original"](../core/camera/mipmaps.md#a-region-of-a-region-readable-original).

  Measured before step 6 on the photograph (4:2:0, interval 4 MCUs, one
  interval per 64 px): five 512 px regions, among them the top-left corner and
  the bottom-right corner, 2.6 to 3.5 ms each, reading about 29 KB of the file
  each, every pixel equal to the whole decode. Indexing the original's 195,075
  intervals took 20 to 63 ms. Measured again with step 6's code: seven 513 to
  514 px regions of one of the user's 16320 x 12240 photographs (4:2:0, interval
  4 MCUs), among them all four corners and the centre, took 0.9 to 2.6 ms each,
  every pixel equal to `read_rgb`; `OriginalRegions::from_bytes`, the region
  test and the index of its 195,075 intervals, took 10 to 12 ms. Those
  regions were one frame each, which `jpeg-decoder` decodes on three threads
  of its own, and inside a kernel that keeps every core busy the same decode
  took about 200 ms (open question 21); since then a region is decoded in
  strips of frames no wider than 128 px, which it decodes on the calling
  thread, about 13 ms a 514 px tile on one idle thread
  ([core/camera/mipmaps.md § "A region of a region-readable original"](../core/camera/mipmaps.md#a-region-of-a-region-readable-original)).
- **Every other original** is read from its `-mip0`, by the tile decode of
  a mip level file above, and never from the original.

Neither path holds a whole level 0 in memory, so no reader of an out-of-core
photograph's level 0 holds more than the tiles it asked for.

## The caches in memory

`PhotographCache` keeps its job and its key, and hands out one kind of
answer for every photograph, whatever its size: a `TiledPhotograph`.

**Tiled photographs, for every photograph.** `get`, `peek`, `peek_state` and
`get_many` answer a photograph with an `Arc<TiledPhotograph>`: its layout, its
coarsest levels when it has mip level files, and a handle to the cache's
photograph tiles, and no other pixels. The cache never builds or keeps a whole
pyramid of any photograph, and every reader, the display and every kernel,
reads pixels through photograph tiles: the panels draw them, and the kernels
read view-region pyramids assembled from them
([view-region-pyramid.md](view-region-pyramid.md)), each render computing from
its warp map the rectangle of each level it reads. The out-of-core threshold
decides only whether a photograph's levels are stored as mip level files; it
does not decide how pixels are read. A kernel's `ProjectedImage` holds the
`TiledPhotograph`, so the kernels' loops have one path. When an out-of-core
photograph's mip level files are missing or stale, the reader that finds it so
builds the mipmap itself (§ "Building a mipmap in the slot"), so a kernel
never reads levels that are not the files' pixels.

**Tiles of an in-core photograph.** An in-core photograph has no mip level
files. Its tiles are made by decoding the photograph whole in its slot (as
`get` decodes it today), building its pyramid to
`full_levels` depth, and cuts every level into photograph tiles with their
one-pixel borders. The tiles the request asked for go into the tile cache as
used; the other tiles of the decode go in as *unused*, the first entries
evicted when the budget needs room, oldest first, and the whole decode is
dropped; an unused tile becomes used when a request reads it. So the
tiles a reader touches stay, and the rest of a decode stays only while nothing
needs the room, which lets the next render in the same photograph find its
tiles without decoding again. Who decodes:

- **A kernel's read phase.** `get_many` decodes its in-core misses (the
  photographs none of whose tiles are cached) into tiles during its read
  phase, in groups sized by the build budget with reservations made when the
  group is decided, exactly as it makes mipmaps, and keeps each decode's tiles
  as unused, so the kernel's renders find them. `get` does the same for its
  one photograph, on its own thread. A render decodes a photograph only when a
  tile it needs was evicted while the kernel ran.
- **A panel.** A panel starts one decode task per in-core photograph, with
  the tiles it is missing, never one per tile. The task claims the slot
  without waiting: if another caller is decoding the photograph, the task
  ends at once and the panel peeks again on a later frame, and a second panel
  that needs the same photograph while its decode runs finds it in
  `PhotographRequests` and peeks on a later frame too. So no panel task waits
  on a slot. A panel's decode reserves with `try_reserve_build`, which refuses
  a decode that does not fit under the build budget (below); a refused
  request is tried again on a later frame. The task reserves first and
  claims the slot only once its reservation is granted, giving the
  reservation back when another caller is decoding in the slot, so a slot a
  panel claims is always decoded, and a kernel caller that waits on it reads
  the tiles it made.

A decode of a photograph whose tiles were evicted costs the whole decode
again: about 0.05 s at 3840 x 2160 and less for
smaller photographs (estimated from the 85 `dino_dog_toy` decodes at 2040 x
1536 taking 0.9 s from one thread, plus the reduction), and 0.8 to 1.7 s for a
200-megapixel photograph in a workspace without `mipmap_options`. The budget
and the bulk kernels' point order keep this rare: bulk kernels take their
points by the image of their reference observation, then by that
keypoint's row and column (the companion draft's § "Memory" and its open
question 3), so the renders of one photograph
follow each other while its tiles are cached. Each whole decode reserves its
estimated bytes on the build count (below) like a build, so the decodes in
progress are bounded with the builds.

**Tiles from an array.** A caller that has the pixels already, such as a
Python binding given numpy arrays, makes a `TiledPhotograph` with
`TiledPhotograph::from_image`, which cuts the image's pyramid into tiles the
source owns, outside any cache, so kernels read arrays through the same path
and get the pixels they got from whole pyramids.

**Building a mipmap in the slot.** The user has decided that making a
photograph's mipmap works exactly like decoding a photograph in today's
`PhotographCache`, with no builder object or queue of its own.

- **On the caller's thread, inside the path's slot.** A `get` that finds an
  out-of-core photograph's files missing or stale makes the mipmap itself, on
  its own thread (a job thread, a rayon worker, a Python thread), and every
  other caller asking for the same path waits on the slot and then reads what
  it made. `get_many` makes its misses in parallel across photographs on the
  rayon pool, as it decodes them today. Each mipmap is made serially inside its
  slot: decode, region test, reduction, and each encode and write in turn,
  with no rayon inside, for the reason
  [photograph-cache.md](../core/camera/photograph-cache.md) § "Implementation
  notes" gives for decodes (a worker that waited in a nested join inside a
  slot could steal another item for the same path and deadlock on itself).
  `jpeg-encoder` and `jpeg-decoder` use no rayon, so the whole build is safe
  there. An in-core photograph's whole decode into tiles runs in its slot the
  same way. A kernel caller that finds the slot being made waits on it; a
  panel's decode task (`try_decode_tiles`) claims the slot only when no other
  caller is decoding in it and otherwise ends at once, so it never waits on a
  slot.
- **Where builds run in the viewer.** Only inside background tasks: the
  quiet task *Build mipmaps* and the kernels of other tasks, each on its task's
  job thread and on the rayon workers its `get_many` makes its group on, as
  `get_many` decodes photographs in parallel today. No panel starts a build
  or waits on a slot: a panel that needs a photograph whose mipmap is not
  current adds it to *Build mipmaps*' list (§ "Readers on the GUI thread"), so
  no rayon task a panel starts, and never the GUI thread, makes a mipmap.
- **Cancellation.** The slot is no longer a `OnceLock`, which cannot be
  abandoned once a thread has started to fill it. It is a mutex over the
  slot's state (empty, being made by one caller, or made, with its result or
  remembered failure) and a condition variable. A caller that finds the slot
  empty marks it as being made and makes the mipmap; one that finds it being made
  waits on the condition variable with `Condvar::wait_timeout` of 50 ms,
  polling its own
  `Progress` for cancellation, and stops with `Cancelled` if it is cancelled,
  leaving the build to its maker. The maker polls its `Progress` between the
  build's stages (after the decode, after the region test, after the
  reduction, and after each level's encode and write), so a cancelled maker
  stops within one stage, at most one decode (0.8 to 1.7 s at 200 MP) or one
  encode. It writes nothing further, removes its temporary file, sets the slot
  back to empty and wakes the waiters; the first waiter to take the lock marks
  the slot as being made and makes the mipmap itself, from the start, and the
  others go on waiting. The levels already written by the cancelled maker are
  complete files: the next maker finds those current and makes only the
  rest. A slot is never left marked as being made by a caller that has
  stopped, a panic included: a guard empties the slot and wakes the waiters
  when the maker unwinds. An in-core photograph's decode into tiles uses the
  same slot and the same cancellation, between the decode and the cutting.
- **Progress.** A build is part of the `get` or `get_many` that makes it and
  reports through that caller's `Progress`: a `get` opens phases *decode*,
  *reduce* and *encode* (the last counting levels); `get_many` counts the mipmaps
  it made and the photographs it read in its own phase, under the name its
  caller gives it (*build mipmaps*, *decode images*), with a note
  (*3 of 17 read, 2 mipmaps made*). Nothing on the cache reports on its own:
  there is no reporter, queue or status object.
- **Memory while building or decoding.** A build, and an in-core photograph's
  whole decode into tiles, holds a whole decode, about 1 GB at 200 MP and
  about 40 MB at 3840 x 2160 (5 bytes per source pixel, estimated), so
  `get_many` does not start all its misses at once. It first resolves each
  path's layout (a header read, the region test and a record check) to know
  which are misses and their sizes, then makes those misses in groups, one
  `par_iter` per group, the groups one after another, with the group's size
  set by memory as `default_concurrent_images_for_dimensions` sizes SIFT's:
  the number of whole decodes that fit in the cache's **build budget**
  (`PhotographCacheOptions::build_budget_bytes`, by default
  `default_build_budget_bytes()`, a quarter of physical memory), less what
  other callers of the same cache have reserved, at most the rayon pool's
  size and at least one. **The count belongs to the `PhotographCache`
  instance**, an atomic in it, not to the process: every caller that shares a
  cache shares its count, and two caches in one process do not coordinate.
  How many caches a process makes is the application's choice: SfM Explorer
  holds one for its session (`AppState::photographs`), a CLI command makes one
  for its run, and a Python script that holds one cache shares its count
  across its calls. A caller **reserves at the moment it decides**:
  `cache.reserve_builds(bytes_per_build, wanted)` adds the
  group's estimated bytes to the count with one compare-exchange loop,
  granting as many of the `wanted` decodes as fit under the budget, and returns
  a guard that gives each decode's bytes back as it ends and the rest
  when it is dropped. The decodes do not add to the count again. A share
  belongs to the caller that makes a mipmap or decodes, not to one that waits:
  `get` reserves its one decode when it marks the slot as being made, whether
  the slot was empty or a cancelled maker has just emptied it, and holds
  nothing while it waits; a `get_many` member that finds its slot being made
  by another caller keeps its share while it waits, since it may take the
  build over, and gives it back when it reads the mipmap or stops. A maker
  that is cancelled, fails or unwinds gives its share back through the guard
  that empties its slot. So two callers sizing a group at the same moment
  cannot both take the same free bytes: the second sees what the first
  reserved. Nothing waits on the count; it is not a semaphore. A kernel
  caller, reserving with `reserve_builds`, that finds the budget used up is
  still granted one decode, so that it makes progress, and that one decode is
  one way the count goes past the budget: by at most one whole decode per
  kernel caller reserving at that moment. A panel's decode reserves with
  `try_reserve_build`, which is refused when the decode does not fit while
  anything else is reserved; with nothing reserved it is granted alone, so a
  panel decodes one photograph at a time under a build budget of 0 and a
  photograph larger than the whole budget is decoded when the count is idle
  (decided by the user, revising the companion's open question 8). That is
  the other way the count goes past the budget, by the one decode.
- **What bounds builds and decodes from independent callers.** The bound is
  per cache. Every build and every whole decode is reserved on the cache's
  count before it starts, so the count is always complete. Kernel callers
  reserve with `reserve_builds`, which grants one decode even when the budget
  is used up, so that a kernel always makes progress: `get_many` for its
  group, `get` for its one photograph, and a render for a photograph whose
  tile was evicted during the kernel. Panels reserve with `try_reserve_build`,
  which refuses a decode that does not fit under the budget, so panel
  decodes stay within the budget and a refused one is tried again on a
  later frame. The count therefore goes past the budget only by the kernel
  callers that reserved while it was full, one whole decode each. A read
  phase is one such caller (its group is granted within the budget, and only
  the one granted past it can overshoot); the renders are the others, at most
  one per thread that renders, since a render decodes only a photograph whose
  tile was evicted during the kernel. A caller that wants several photographs
  asks for them with one `get_many`, never with `get` from many workers of a
  parallel loop (no caller does today; the reference picks call `get` one
  photograph at a time on their job thread). So the worst case is the build
  budget plus one whole decode per thread that renders or reads: in the
  viewer, with the 32 rayon threads and the two background queues' job
  threads, the build budget plus 34 decodes, about 1.4 GB at 3840 x 2160, and
  about 34 GB at 200 MP in a workspace without `mipmap_options`, where every
  photograph is in core and every decode is whole; avoiding that second
  figure is what the out-of-core tier is for, since an out-of-core
  photograph's tiles decode alone, a few milliseconds and under a megabyte
  each, and only its mipmap build holds a whole decode, made in a read phase
  or by *Build mipmaps*. Panel decodes add nothing past the budget. Mipmaps
  are made only in the read phases of background tasks, at most one running
  per queue, so at most two of the decodes past the budget are mipmap
  builds, 2 GB at 200 MP; the build budget is 16 GiB on the 64 GB development
  machine and 4 GiB on a 16 GB machine. The figures are the worst case, every
  rendering thread finding an evicted photograph at once; a kernel whose
  photographs' tiles fit the cache's budget decodes each photograph once, in
  its read phase. A CLI command uses one cache for its run, so its bound is the
  build budget plus one decode per thread of its one kernel call (33 on the
  development machine); a Python script that holds one cache gets the same,
  plus one decode per extra thread calling it at once. A program that makes
  several caches gets one such bound per cache, by its own choice. These are
  on top of the cache's budget, and are estimates from the buffer sizes.
- **No build on the GUI thread, and no tile task that waits on one.** The
  panels never call `get` for an out-of-core photograph, on the GUI thread or
  on the rayon pool, whatever the budget, and ask for its tiles only once its
  files are current (§ "Readers on the GUI thread"), so a tile task never
  meets a build. A panel's decode of an in-core photograph is one task per
  photograph that claims the slot and reserves without waiting
  (`try_decode_tiles`), so no tile task a panel starts waits on a slot or on
  the count. No panel decodes on the GUI thread either.

`get_tile`, `view_region_pyramid` and `get_level` never make a mipmap. For an
out-of-core photograph they are asked only once its layout is `Known` (its files
current), and decode only tiles of current files, milliseconds each (`get_level`
at most a few hundred milliseconds for a level of 48 tiles). For an in-core
photograph a miss decodes the whole photograph, up to 0.8 to 1.7 s at 200 MP, so
they and `try_decode_tiles` take the caller's `&Progress`, which they poll for
cancellation as `get` does, answering `TileError::Cancelled`, never remembered,
with the reservation given back. For an in-core photograph, `get_tile` called
from a render entry point on a kernel's behalf is a kernel's read: a miss
decodes the photograph into tiles, as above, about 0.05 s at 3840 x 2160,
reserving with `reserve_builds` and waiting on the slot if another caller holds
it; inside a kernel this happens only for a tile evicted after the read phase
decoded its photograph. Panels read an in-core photograph's missing tiles with
`try_decode_tiles` instead. A kernel whose files go stale after its read phase
found them current (another process rebuilt them, or the original changed) gets
`TileError::Changed` from its next tile read and fails as § "Files that change
while a kernel runs" says, rather than rebuilding inside a render and going on
with levels of two versions.

**Files that change while a kernel runs.** A tiled source records, for each
file it has read, the file's record (for an in-core photograph, the
original's length and modification time). When a tile is decoded again after
it or its file's bytes were evicted, the file is read again and its record
compared with the one recorded: equal (the same source stamp, quality, region
answer, encoder and sfmtool version, so the same deterministic output) and
the kernel goes on; different, and the read fails with `TileError::Changed`,
naming the file, which the kernel reports as its failure (*The mipmap of
images/IMG_0001.JPG changed while Refine all normals ran; run it again*). The
source does not pin the bytes of every file for the kernel's whole run, since
for hundreds of photographs that would be the very memory this design removes,
and a kernel that mixed two versions of a level would give results no single
version gives.

**Photograph tiles.** Keyed by path, level, column and row; each holds the
512 x 512 core and a one-pixel border taken from the neighbouring tiles'
pixels, or the edge pixel repeated at the image's edge (514 x 514 x 3 = 793 KB
at most). The border is what lets the display draw tiles side by side with
linear filtering and no seam. Where a tile comes from:

| Photograph | Level | Source |
|------------|-------|--------|
| in core | any | cut from a whole decode of the original and its pyramid, made when a tile of it is missing |
| out of core | 1 to `K` | the level's mip level file, a tile decode from its bytes |
| out of core | below `K` | the levels computed from level `K`, kept whole with the coarsest |
| out of core, region-readable original | 0 | a region decode from the original's bytes |
| out of core, other original | 0 | the `-mip0`, a tile decode from its bytes |

A file's bytes (10.8 MB for level 1 here, 30 to 43 MB for an original or a
`-mip0`) and its restart index are an entry too, read once and kept while
tiles of it are asked for, so a session reads one version of each file.

**One budget.** The tiles and file bytes are the cache's only entries, under
one budget, `PhotographCacheOptions::budget_bytes`: by default
`default_budget_bytes()`, a quarter of physical memory clamped to 1 to
16 GiB, overridden by `SFM_EXPLORER_PHOTOGRAPH_CACHE_MB`, the budget and the
variable the pyramids had. Eviction is least recently used, with two rules:
the unused tiles of an in-core decode go first, and the coarsest level of every
out-of-core photograph whose mip level files have been read, with the levels
below it (at most one tile, 0.78 MB for this photograph), goes only after every
other entry. The decode that made a request's tiles drops none of them before
the next decode when the request is `get`'s, or a render's read through a
photograph `get` answered of a photograph under half the budget (step 13e5,
the companion's open question 12). A tile or region held by a running render
stays alive until the render drops it, as a pyramid does today. A budget of 0
keeps nothing: each request is decoded and handed to its asker, whose texture
or render keeps it.

**States, and what a caller on the GUI thread may do.** `peek_tile` and
`layout` answer without reading or waiting:

```rust
pub enum PeekedLayout {
    Known(Arc<PyramidLayout>), // header read, and for out of core files current: tiles may be asked for
    Building { coarsest: Option<Arc<CoarsestLevel>> }, // mipmap missing, stale or being made; the coarsest once published
    Unreadable(GetError),       // the original or the workspace file cannot be read; the error says which
    NotKnown,                   // not checked yet
}
pub enum PeekedTile { Ready(Arc<PhotographTile>), Building, Unreadable, NotReady }
```

`Known` is answered for an in-core photograph once its header is read, and for
an out-of-core one only once its files have been checked current (or its
in-memory equivalent made); `peek_tile` answers `Building` for any tile of a
photograph that is `Building`. A tile is asked for (`get_tile`) only for a
photograph that is `Known`, so a tile request never waits on a build: it
decodes a tile of a current file (milliseconds) or decodes an in-core
photograph into tiles (about 0.05 s at 3840 x 2160), a panel's through
`try_decode_tiles`, which ends at once rather than wait on a slot or on the
build budget. If the files change under
a `Known` out-of-core photograph, `get_tile` finds the record stale, answers
`TileError::Changed` without waiting, and the photograph goes back to
`Building`, so the panel adds it to *Build mipmaps*' list again. A mipmap is
made only inside a `get` or `get_many` of a background task, never on the GUI
thread and never by a panel.

**Memory, under this design.** What a kernel holds is the view-region
pyramids of the renders it has in flight or keeps for reuse, plus the cache's
budget, whatever the number or size of the photographs, in core or out. A
render's region pyramid at patch resolution `R` with a footprint that is not
stretched is about `(R + 4)^2 x 3` bytes at its level and a quarter of that at
the next (anisotropic blending), so 2.9 KB at `R = 24` and 17 KB at `R = 64`,
and about 2.25 times that when it is kept for reuse, grown by
`REGION_REUSE_MARGIN` (a quarter of the region's width and height on each
side, [core/camera/view-region-pyramids.md](../core/camera/view-region-pyramids.md) § "Theory"). A stretched footprint, a grazing view, or the
`Bilinear` sampler reading level 0 over a large footprint, can make one
render's bounding box far larger; the companion draft bounds that with a
second sampling path and one region pool per kernel call, made by the
kernel's caller and passed in, for everything copied (§ "Bounding a render's memory" there). A kernel's patches do not map
one to a worker (*Refine all normals* runs a parallel loop over points whose
normal kernels run parallel loops over views, and a worker waiting in an
inner loop takes other points from the outer one), so the bound is not per
patch. It has two parts. The copied regions of one kernel call, all its
patches together, reserve from a pool of 3.1 MB per rayon thread, 99 MB on
the development machine's 32 threads; a render that cannot reserve takes the
tile path and keeps nothing. The pinned tiles are 16 (12.7 MB) per render
running, and a thread runs one render at a time (a render's own rows never
go parallel on a worker, `parallelize_rows` in `camera/warp_map.rs`, and a
render on the tile path keeps its rows serial on every thread), so at
most 12.7 MB per thread that renders: the 32 rayon threads, the calling job
thread, and in the viewer the other queue's job thread and the GUI thread.
One kernel call on the command line holds at most 99 MB + 33 x 12.7 MB, about
520 MB, on top of the budget; the viewer, with at most two kernel calls at
once (one per background queue) and the renders of Track View and the
Track Table, at most 2 x 99 MB + 2 x 3.1 MB + 35 x 12.7 MB, about 650 MB. These figures hold for in-core
photographs as for out-of-core ones, since both are read through the same
tiles. A build or a whole decode renders nothing and pins no tiles, so it does
not change the count; its own memory is the build budget's bound in
§ "Building a mipmap in the slot", separate from this one. The common case is
a few hundred kilobytes, whatever the number of views or photographs
(estimates from the sizes).

| Case | Today (whole pyramids) | This design |
|------|------------------------|-------------|
| 17 x 200 MP, *Find matches by geometry* over the views the cull keeps (up to 151), `R = 24` | up to every photograph, 12.6 GiB for all 17 | up to 151 x 2.9 KB x 2.25 = 1.0 MB of regions kept for reuse (well inside the call's 99 MB pool), plus at most the budget (16 GiB at 64 GB of memory, 2 GiB at 8 GB) |
| 17 x 200 MP, display patch bitmaps at open, one render per point | 12.6 GiB | 32 renders in flight x 2.9 KB, plus the budget |
| 500 x 200 MP, any of the above | 372 GiB; fails | the same as for 17: regions plus the budget |
| Any of the above, worst case (the kernel call's region pool full and every rendering thread on a grazing view) | — | one kernel call: 99 MB + 33 x 12.7 MB, about 520 MB; the viewer with two kernel calls and Track View: about 650 MB; a kernel call that runs the cluster refinement adds evaluation tiles past the pool, at most 8.4 MB per request per rendering thread at the default resolution (280 MB for one level), though a member's tile is the union of its requests at a level and grows as far as its cascade wanders, so that bounds each request, not what one member holds (the companion's § "The members' elongation limit"); on top of the budget |
| 500 x 6000 x 4000, any of the above | 500 x 96 MB = 48 GB | the same |
| 17 x 200 MP, Track View on a 17-image track at `R = 64` | 12.6 GiB | 17 x 17 KB x 2.25 of regions, plus the tiles they are cut from, one to four per level read and one or two levels per view, 13 to 108 MB |
| 368 x 3840 x 2160 (in core), a `get_many` over every frame | 12.2 GB held while its kernel runs | the regions of the renders in flight plus the budget; each frame decoded whole once into tiles in the kernel's read phase (about 0.05 s and 40 MB while it decodes), as many at once as the build budget allows, and again, by a render, only for a frame whose tiles were evicted during the kernel |
| The test data (17 x 270 x 480 to 85 x 2040 x 1536, in core) | 8.8 MB (17 x 270 x 480) to 1.1 GB (85 x 2040 x 1536) of pyramids | regions plus the tiles touched, within the budget; a whole decode is 0.6 to 16 MB and about 11 ms at 2040 x 1536 |

The cost moves from memory to decodes: a render whose tiles are not cached
costs 2.3 to 6 ms per tile it touches out of core, one to four per level it
reads, and a whole decode of an in-core photograph whose tiles were evicted
(about 0.05 s at 3840 x 2160). A bulk kernel whose tiles do not fit the
budget decodes some tiles, or some in-core photographs, more than once; it
takes its points by reference image, then by the reference keypoint's row
and column (§ "Memory" in the companion draft), and its phase notes report the tiles decoded and reused and
the whole decodes, so a run that decodes far more than once per photograph
shows it.

The display holds, for 17 photographs of 200 MP, the visible tiles of one
level (at about one texel per screen pixel a 3840 x 2160 panel covers at most
9 x 6 = 54 tiles, 43 MB) plus the coarsest resident of each photograph
(13 MB). For 368 frames of 3840 x 2160 it reads the tiles of the frame it
shows, decoded into tiles once per frame, and keeps those it touched while
the budget allows.

**Track View's requests.** Track View asks for the layouts of a track's
photographs and then for the tiles its cells' regions cover, for every
photograph, in core or out; a frame's tiles are a few megabytes, well within
the budget, so the panel never asks for more than the cache can keep.

## Who makes the mipmaps, and when

- **`sfm sift --extract`** makes them for every out-of-core photograph it
  reads, after the `.sift`, so a solve's workspace has them before the viewer
  opens it. `sfm sift` runs the Python extractors (`sfmtool` and OpenCV
  backends decode with `read_image_rgb`; COLMAP decodes inside COLMAP), and
  the user has decided that the SIFT path and the mipmap build stay separate
  (open question 4): once every `.sift` of the run is written,
  `image_files_to_sift_files` calls `PhotographCache.build_mipmaps`
  (§ "Python bindings") once over the photographs it extracted, on one cache
  of budget 0 the run holds, so its builds share one build count, and that
  call decodes each out-of-core photograph whose mipmap is not current
  itself (built in step 15d, whose note says why after the run rather than
  after each file). So on the command
  line the mipmaps cost a second decode (0.8 to 1.7 s at 200 MP) besides the
  reduction and the encodes (about 0.8 s, plus 1.4 to 2.1 s for a
  `-mip0`, estimated from the parts above). Only the Rust `build_sift_file`
  in [sift/files.rs](../../crates/sfmtool-core/src/features/sift/files.rs),
  behind the viewer's *Build SIFT files* and `pick-images`, makes them from
  the decode it already holds, passing it as `build_mipmaps`'s `decoded`
  argument, since there the decode is in Rust and handing it over is a
  borrow, not a binding's array (built, step 15a, before the extraction).
- **`sfm ws mipmaps PATHS…`** makes them explicitly for the photographs under
  the paths (skipping in-core ones and current mipmaps), with `--prune` to delete
  orphans and `--check` to report stale mipmaps without writing. It is the way to
  prepare a workspace whose features came from somewhere else. Built in step
  15e ([ws-mipmaps-command.md](../cli/workspace/ws-mipmaps-command.md)).
- **Any reader of the cache** makes the mipmaps it needs, in any process, as
  § "Building a mipmap in the slot" says. On the command line and in Python,
  `ImagePyramidSet.read(recon, progress=…)`, `embed-patches`,
  `sfm cluster-patches` and `sfm web-export` read through a cache whose
  `get_many` makes every missing mipmap in its phase, counting mipmaps through the
  caller's `Progress` (a `ProgressCounter` from Python), before the work
  starts; `sfm ws mipmaps` calls `PhotographCache.build_mipmaps`, which
  calls `get_many`, over every photograph under its paths (built, step
  15e).
- **In the viewer**, mipmaps are made by one cancellable quiet task, *Build
  mipmaps* (`Operation::BUILD_MIPMAPS`, a fifth quiet source beside those of
  [background-tasks.md](../gui/background-tasks.md) § "Quiet tasks"), like
  the quiet tasks Track View and the Track Table use. It runs on the **fast
  queue**, with the other quiet tasks: on the slow queue it would wait behind
  a bundle adjustment or a *Refine all normals* of minutes while a panel shows
  its spinner. It gives way to anything asked for in its queue under the
  quiet-task rules: its cancel flag is set, each build in progress stops
  within one stage and leaves its slot empty, and it is queued again behind
  the task that displaced it, losing at most the stage each build was in.
  In the `QuietRank` order it comes after the camera model proposal's job and
  before the evaluations, since a panel is waiting on it. It does not lock
  its node, runs on a locked node (it reads photographs, not the node's
  version), and pushes no version: it writes mip level files and fills the
  session cache. Its row in background-tasks.md's operations table is
  *Build mipmaps | fast | no; it pushes no version (it writes mip level files
  and fills the photograph cache), and it runs on a locked node | yes*.

  Its source keeps a list of photographs shared with the viewer. **Each run
  of the task makes one chunk and ends**: its job, on its job thread, takes
  one chunk from the front of the list and calls `get_many` on it with its
  own `Progress` and collector, so the parallelism is `get_many`'s, and when
  the job ends with the list not empty, the source queues the task again.
  A chunk is the photographs the build budget lets the cache make at once
  (as many as `reserve_builds` would grant from an empty count, at most the
  rayon pool's size), at least one, so a run takes about one build, 2 to 4 s
  at 200 MP. Because a quiet task that is running gives way only to a task
  someone asked for (§ "Quiet tasks" applies `QuietRank` to waiting tasks),
  ending after each chunk is what lets `QuietRank` decide between chunks:
  a live evaluation, the Track Table's columns or picks, or a camera model
  proposal's job waiting in the fast queue starts before the next chunk
  rather than after the whole list. A job that is cancelled puts the
  photographs of the chunk it had not finished back at the front of the
  list, so the run queued again behind the displacing task makes them
  first. When a reconstruction opens, the list is
  the reconstruction's out-of-core photographs whose files are missing or
  stale (found by a `stat`, the original's header and markers for the region
  test, and one small read each): the selected image first, then the images
  the Image Browser strip shows, then the rest in image order, behind the
  photographs a panel has queued and ahead of an earlier open's. Closing a
  node takes the photographs no node left in the scene has off the list
  (§ "Open questions" 19). A panel that needs a photograph whose mipmap is
  not current **adds it to the front of the list**: the next run takes it in
  its chunk, the running chunk being left to finish, and when no run is
  running or waiting one is queued, under the quiet-task rule that a source
  keeps at most one waiting entry. While it
  runs, the Background Tasks panel's quiet line shows it as it shows every
  quiet task, as *Building mipmaps: {done} of {total}*.
- **A kernel** inside another background task that needs an out-of-core
  photograph whose files are not current makes the mipmap itself, as part of
  its task, in its read phase (its `get_many`, on its job thread and the
  rayon workers `get_many` uses), counted in the phase's note (*3 of 17 read,
  2 mipmaps made*); if *Build mipmaps* is making that photograph, the kernel
  waits on the slot on its job thread rather than start a second build. It
  does not read the exact levels in the meantime.

**Readers on the GUI thread.** None of them makes a mipmap, decodes a
photograph on the GUI thread or waits. Each asks `layout` and acts on the
state.
*Queue the photograph* means: add the photograph to the front of *Build
mipmaps*' list, once per path (the viewer remembers the paths it has queued,
as `PhotographRequests` remembers its decodes, and forgets a path when the
task reports its mipmap made or failed, so a photograph that goes back to
`Building` is queued again), and draw a loading spinner
with *Building mipmap…* until the photograph's layout is `Known`, repainting
when the task reports it made. Tile reads go to the rayon pool, only for a
photograph that is `Known`: for an out-of-core photograph one task per missing
tile, a tile decode of a current file, milliseconds; for an in-core one, one
decode task per photograph with all the tiles the panel is missing (about
0.05 s at 3840 x 2160), which claims the slot without waiting and reserves
with `try_reserve_build`, ending at once when another caller holds the slot
or its reservation is refused, so the panel peeks again on a later frame and no
rayon task a panel starts waits.

| Reader | `NotKnown` | `Building` | `Known` |
|--------|------------|------------|---------|
| Image Detail | asks for the layout once (a header read, the region test and a record check, on the rayon pool) and draws *Loading image…* | queues the photograph, draws the coarsest level if the build has published it and the spinner with *Building mipmap…* otherwise | peeks the visible tiles and asks for the missing ones |
| Track View's patch tiles, cluster-stage tiles, context and crops; the Track Table's crop column and hover | the same layout request; the cell draws its loading state | queues the photograph; the cell draws the spinner with *Building mipmap…* | computes the render's regions, peeks their tiles, asks for the missing ones, and renders when all are present |
| The 3D background | the same layout request; nothing drawn | queues the photograph; draws the coarsest level if published | asks for its level's tiles with `try_decode_tiles` on the rayon pool and builds the level from the tiles that call returns, at every budget, and draws the coarsest until it lands |
| Display thumbnails | built in the open job, off the GUI thread: a photograph whose coarsest mip level file is not current gets the grey placeholder row | the row stays the placeholder | a placeholder row whose photograph *Build mipmaps* has answered is filled, on the rayon pool, from the coarsest mip level file, or from a whole decode when the mipmap is still not current; one made during the open is filled by the open |

A build publishes each reduced level to the slot's `Building` state as it is
encoded, coarsest first, so a panel shows a photograph's coarsest level about a
second after the build starts, before its finer levels and its `-mip0` are
written.

**A budget of 0.** `display_photograph` with `SFM_EXPLORER_PHOTOGRAPH_CACHE_MB=0`
called `get` on the GUI thread, since nothing it decoded would be kept for the
next frame; Image Detail stopped calling it in step 13c2, the 3D background in
step 13c3, Track View in step 13d1, and the Track Table and Measurement View's
endpoint crops in step 13d2, which deleted it. No panel decodes on the
GUI thread in this design, at any budget: with a budget of 0, a tile decoded
on the rayon pool is handed to the panel that asked for it, whose texture
keeps it, the cache keeps nothing, and an out-of-core photograph is still
queued for *Build mipmaps*.

## Display

### Image Detail

The panel stops making a texture of the photograph and draws photograph tiles.
Step 2 built this from the cache's in-memory pyramids, as
[gui/multi-panel-image-browser.md](../gui/multi-panel-image-browser.md)
§ "Image Detail" describes, with the tiles asked for from the cache
(`peek_tile`), the next image's tiles asked for a frame ahead and the two
`get_image_detail_view` fields built by step 13c2, except the `Building`
state, which comes with step 14:

- **Which level.** With `s` the panel's physical screen pixels per level-0
  pixel (`effective_scale` times the pixels per point), the level is
  `round(log2(1 / s))` clamped to the photograph's levels, so a displayed
  texel covers between 0.71 and 1.41 screen pixels; at `s ≥ 0.71` it is level
  0, full resolution (the original's pixels, or `-mip0`'s). The coordinate transform, the overlays and
  the bench layer keep working in level-0 pixels; level `l`'s tile at column
  `c` and row `r` is drawn over level-0 pixels `c·512·2^l` to
  `(c+1)·512·2^l` (and the same for rows), clipped to `w_l·2^l`, so the
  dropped columns at the right and bottom edge are less than one texel wide.
- **Which tiles.** Those that intersect the panel's visible rectangle, asked
  for with `peek_tile` once the photograph's layout is `Known`; a miss
  requests the tile on the rayon pool (a decode of a current file,
  milliseconds of work) and repaints when it lands. While the photograph is
  `Building` the panel requests no tiles (§ "Readers on the GUI thread").
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
  coarsest resident level is drawn at once if its mipmap or tiles have been
  read this session, then its tiles at the current view. Until the new
  image's overview tiles have landed, the panel keeps drawing the image it
  showed before (built in step 13c2, decided by the user). While animation plays,
  the next image's tiles for the current view are asked for one frame ahead.
  The view persistence rule (§ "View persistence across image, reconstruction
  and panel changes") is unchanged; it computes `display_size` from the
  photograph's level-0 size, which the panel reads from the header (or the
  camera) rather than from a texture.
- **Zoom limit.** The limit becomes `max(32, 8 / fit_scale)` times the fit,
  where `fit_scale` is the panel points per level-0 pixel at the fit, so a
  level-0 pixel can always be enlarged to 8 points, which is 8 screen pixels
  at one physical pixel a point and more on a denser display (open question
  7). Points are the unit the panel's view arithmetic and
  `set_image_detail_view` work in; the level rule above alone reads physical
  pixels.
  The fixed limit of 32 times the fit before step 2 reached only 0.78 screen pixels per
  level-0 pixel for a 16320 px photograph in a 400 px panel, so the
  original's pixels could not be enlarged at all; for a photograph that fits
  a panel at 1/4 or larger, the limit stays 32.
- **Agents.** A `screenshot` of the panel shows what the frame drew, coarser
  tiles included. `get_image_detail_view` gains `level`, the level the panel
  draws, and `tiles_pending`, how many visible tiles are not yet drawn at that
  level, so an agent waits for `0` before a screenshot it means to judge pixels
  by ([gui/mcp-server.md](../gui/mcp-server.md) § "The Image Detail view").

### The 3D view's camera-view background

The background becomes one texture of one level, made off the GUI thread.
Built (step 3) from the level of the cache's whole in-memory pyramid, asked
for with `display_photograph`; the level rule, the texture coordinates, the
alpha and the remembered failure are as below. Until a photograph lands, the
previous image's photograph stays on its own mesh (nothing is drawn before
any has landed), since there is no coarsest level to draw before the tiled
cache. Step 13c3 (built) swapped the source for the level built from the
photograph's tiles, through two closures the app passes to
`upload_bg_image`, `display_layout` and `display_level`, as
[gui/camera-views.md](../gui/camera-views.md) § "Photograph background"
describes:

- **Level.** The coarsest level whose longer side is still at least the
  viewport's longer side in physical pixels, but no finer than the first level
  whose longer side fits `max_texture_dimension_2d`. At 3840 px of viewport
  that is level 2 (4080 x 3060) of the 200-megapixel photograph; at 1920 px,
  level 3.
- **Source.** The level assembled from the photograph's tiles, once the
  photograph is `Known`, for every photograph. On the rayon pool the
  background asks for the level's tiles with `try_decode_tiles`, asking
  again on a later frame when the answer is `Busy`, and builds the level from
  the tiles that call returns, at every cache budget, never calling
  `get_level`. So no panel task waits on a slot or on the build count, and
  that rule has no exception (decided by the user, the companion's open
  question 9). The coarsest level is drawn
  until it lands once the tiled cache can hold it apart from the level
  (steps 13e and 14); until then the coarsest level comes from the same
  decode as the level drawn, so the previous photograph stays meanwhile, as
  before step 13c3. A photograph that cannot be read is remembered as
  a failure, so it is not read again on every frame.
- **Texture coordinates.** The mesh's texture coordinates are scaled by
  `width / (w_l·2^l)` and `height / (h_l·2^l)`, so the dropped edge columns do
  not shift the picture.
- **Alpha.** The background reads RGBA today, but its pipeline draws with no
  blending (`blend: None` in `scene_renderer/pipelines/bg_distorted.rs`, and
  the shader returns the sampled colour), so the alpha has no visible effect.
  It reads the cache's RGB, with alpha 255, for every photograph, and nothing
  on screen changes; mip level files having no alpha costs nothing here.

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
an out-of-core photograph it is made from the coarsest mip level file rather than from
a whole decode, which turns 0.8 to 1.7 s into a few milliseconds. Its pixels
differ slightly from a thumbnail of level 0; it is for display only. One whose
mipmap is not current when the reconstruction opens is the grey placeholder
until *Build mipmaps* has answered it, and is then filled in from that file (§
"Readers on the GUI thread").
Thumbnails written into files (`xform --add-thumbnails`, `.sift`) keep coming
from level 0.

## Every reader of photograph pixels

The readers as they are today (all paths under `crates/` unless written
otherwise), what each reads after this design, and whether its results change.
"Regions" means the reader reads view-region pyramids assembled from
photograph tiles, for every photograph: an in-core photograph's tiles are cut
from a decode of the original, and an out-of-core one's come from levels 1 to
`K` of its mip level files and level 0 of the original if it is
region-readable or of its `-mip0` otherwise. "The stored levels" means the results change for an
out-of-core photograph, whose levels 1 to `K` (and `-mip0`) are JPEG files;
an in-core photograph's tiles hold today's exact levels. Each row marked
"regions" moves to view-region pyramids in its own step (steps 9 to 12 and
16), while the cache still hands out whole pyramids, which a moved reader wraps
in a `TiledPhotograph`; once every reader has moved, the cache hands out
tiled photographs only (step 13), so no step leaves a reader without pixels.

| # | Reader | Today | After | Results change |
|---|--------|-------|-------|----------------|
| 1 | Image Detail (`dock.rs` → `display_photograph` → `load_image`) | level 0 as one texture | photograph tiles of the zoom's level; coarsest resident | display only; no texture over 514 px |
| 2 | 3D camera-view background (`bg_image.rs` `load_bg_texture`, from `app.rs`) | whole RGBA, one texture, decoded on the GUI thread | one level fitting viewport and limit, assembled from tiles off the GUI thread | display only; the alpha it reads today is not drawn |
| 3 | Track View's requests (`dock.rs` `cache_bench_track_images`, `cache_selected_point_images`) | a pyramid request per image per frame | the layout, then the tiles each render needs | no |
| 4–6 | Track View's patch tiles, cluster-stage tiles, hover context (`track_view/body/mod.rs` `ensure_tile`, `ensure_context`; `tile.rs`, `patch.rs` → `render_tile`; cluster stage → `sample_member_grid`) | whole pyramid, `peek` | regions, from peeked tiles, read through `render_source_tile` and `sample_source_member_grid` (built in step 9 over the whole pyramids); the *Zoom* cell and the *ZNCC* hover read no photograph | the stored levels |
| 7–9 | Track View's crops and their hover, the Track Table's crop column and hover (`patch_pictures/crop.rs` `picture`, `track_table_panel.rs` `crop_picture`, `track_table_panel/pictures.rs` `crop_context`) | one level of the pyramid, `level(0)` for the size | the crop's rectangle of the same level, from peeked tiles (the companion's § "Readers that sample through their own code"); the size from the layout | the stored levels |
| 10 | Track Table reference picks (`track_table_panel/picks.rs` → `render_reference` → `render_view_tile`, `clipped_share`) | `get`, whole pyramid; level 0 inside the outline; `level(0)` for the size check | regions, and `clipped_share` walks the level-0 tiles under the outline; each pick still fetches its views with one `get` per observation, and makes one region pool for its one `render_reference` call over all of them, so a point's views share one pool; the size check from the layout | the stored levels, and `clipped_share` reads `-mip0` for an original that is not region-readable |
| 11–13 | Display patch bitmaps at open, conversion to embedded patches, `sfm web-export` (`display_bitmaps.rs`, `state/edits.rs` `convert_to_embedded_patches_job`, `web_export.rs`) | `get_many`, full depth, every photograph held at once; web-export through a cache of budget 0 before 16e, and since 16e through the cache its caller passes (the command's has the default budget) | `get_many` answers tiled photographs; regions; web-export makes missing mipmaps in the workspace | the stored levels |
| 14–18 | The bench's steps and searches, live evaluation, *Find Nearby Tracks*, *Create Track Here*, *Add image to tracks*, *Refine all normals*, *Refit all points* (`bench.rs`, `state/edits.rs` `fetch`, `bench/nearby_tracks.rs`, `bench/track_at_pixel.rs`, `add_image_to_tracks.rs`, `state/edits/refine_all_normals.rs`, `refit_all_points.rs`) | `get_many`, levels per sample or per grid, `clipped_share` on level 0; the normal kernels' renders, the grown plane's own sampler and the cluster refinement's `LevelTile` copies | regions, `clipped_share` as in row 10; the grown plane's reads through `read_source_samples` and `LevelTile` through `read_source_rect` (the companion's § "Readers that sample through their own code") | the stored levels |
| 16a | *Find Nearby Tracks*' grey reads (`read_grey`, `bench/nearby/grey.rs`) | each read blurs the level-0 rectangle it samples, plus the blur's 4 px, through `read_source_rect` over the whole pyramid `from_pyramid` wraps (built, step 12); before step 12, level 0 of each view, blurred grey `f32`, whole: 800 MB per 200-megapixel view | the same rectangles, from the cache's tiles | level 0 from `-mip0` for an original that is not region-readable |
| 19 | The viewer's cluster-patches build (`cluster_patches.rs` `read_photographs`) | `get_many` through the cache (built, 11b), full depth, every photograph held at once | `get_many` answers tiled photographs; regions | the stored levels |
| 20 | Display thumbnails when none is stored (`display_thumbnails.rs` → `display_thumbnail_row`) | whole decode, resized to 128 | in core unchanged; out of core from the coarsest mip level file | display only |
| 21 | `.sift` building (`features/sift/files.rs` `build_sift_file`) | whole decode | whole decode of the original, then the mipmap from it | no |
| 22 | Image readings at save (`reconstruction/image_readings.rs` `gather_image_readings`) | whole decode of every photograph with no `.sift` row, with the number decoded at once bounded by memory as SIFT's is (built in step 4) | unchanged | no |
| P1 | `sfmtool.fileio.read_image_rgb` / `read_image_rgba` (`sfmtool-py/src/fileio/image.rs`) | whole decode | unchanged: they read files, not pyramids | no |
| P2 | `ImagePyramidSet(recon, images)` and the kernels' `images` lists (`sfmtool-py/src/patches/views.rs`) | exact pyramids of the arrays passed | tiled photographs cut from the arrays (`TiledPhotograph::from_image`), the same pixels; a new `ImagePyramidSet.read(recon)` reads by path through the core cache | no for arrays; through `read`, as rows 11–18 |
| P3 | `embed-patches`, the strip montages, the `xform` steps that read photographs (`src/sfmtool/_commands/embed_patches.py`, `src/sfmtool/strips/`, `src/sfmtool/xform/_apply.py`) | `ImagePyramidSet.read` through one cache per run: `embed-patches` and the `xform` steps (built, 16c), the strip montages (built, 16d) | `ImagePyramidSet.read(recon)` | the stored levels, so they match the viewer |
| P4 | `sfm cluster-patches` (`src/sfmtool/_cluster_patches.py`) | `ImagePyramidSet.read` over the paths the file names, `get_many` through a cache of its own, as #19 reads them (built, 16b); before 16b, `read_image_rgb` arrays, full depth, every photograph held at once | through the same reader as #19 | the stored levels |
| P5 | The view-tile binding (`sfmtool-py/src/patches/view_tile.rs`) | an exact pyramid of the array passed | a tiled photograph cut from the array | no |
| P6 | Optical flow's pyramids (`sfmtool-py/src/flow/warp.rs`) | a grey pyramid of its own from an array | unchanged; flow reads level 0 from its array | no |
| P7 | Spherical tile source stacks (`sfmtool-py/src/spherical/tile_source_stack.rs` → `spherical/per_tile_source_stack.rs`) | `remap_bilinear` of the arrays passed | unchanged: they read arrays, not pyramids | no |

The measurement harnesses under `scripts/` (`track_at_pixel/context.py`,
`add_image_to_tracks/harness.py` and `misses.py`, and
`reference_view/measure.py`) read their photographs with
`ImagePyramidSet.read` (built, 16e). `bench_bars/measure.py` is a P2 caller:
it plants rows into arrays of the photographs, so every set it scores,
planted or not, is built from the same arrays with the constructor, and a
reading changes only by the plant. A P2 caller that passes arrays holds them
whole and reads their exact levels; that is the caller's choice of input,
and `read` is the bounded way.

Every reader that reads a pyramid of a photograph now reads the same levels
whether it runs in the viewer or on the command line, and in any session. A
reader of level 0 reads the original's own pixels when the photograph is in
core or its original is region-readable, and `-mip0`'s otherwise.

## Standards

- **Progress.** `build_mipmaps` takes `&Progress`, opens phases for the
  decode, the reduction and the encodes, counts levels and polls for
  cancellation between them; a cancelled build writes nothing further and
  leaves no partial file. `PhotographCache::get` gains a `&Progress`
  parameter, since a `get` can now make a mipmap (2 to 4 s) and already can take
  over a second to decode a large in-core photograph; callers with nothing to
  report pass `&Progress::none()`. `get_many` passes its own down. `get_tile`,
  `view_region_pyramid`, `get_level` and `try_decode_tiles` take one too,
  polled for cancellation only: they never make a mipmap, and a tile of a
  current file takes milliseconds, but an in-core miss decodes the whole
  photograph, up to 0.8 to 1.7 s at 200 MP. `sfm ws mipmaps` reports through a
  `ProgressCounter`. `sfmtool-progress` is used unchanged: a build reports
  through the caller's `Progress` with the phases, counts and cancellation it
  already has, and nothing in this design adds a reporter to it.
- **No panics on caller input.** A mip level file that is missing, cut short, damaged,
  disagreeing with its name, or recording another photograph is stale and is
  made again; an original that cannot be read is an `Unreadable` answer, as
  today; a workspace file with an invalid `mipmap_options` is an error naming
  the key; a photograph with a side of 65536 px or more is refused from its
  header with `GetError::TooLarge`, before any decode. So every file the tier
  writes fits a JPEG frame, and `MipmapError::TooLarge` is only the encoder's
  own refusal,
  reported rather than unwrapped. The region test and the tile decoder's
  slicing are checked against the index, never trusted from the file.
- **Points at infinity.** Unaffected: the design changes where pixels come
  from, not any geometry, and the region of a render is computed from its warp
  map, whose invalid texels (a point at infinity behind the camera, a ray that
  does not reach the patch) take no sample and so add nothing to the region.
- **Logic in Rust.** The naming rule, the record, the region test, the
  staleness test, the workspace object's validation, the slot, the
  making and the reading are Rust in `sfmtool-core`; Python calls bindings.
- **Settings structs.** `MipmapOptions` is the workspace's `mipmap_options`
  object; `PhotographCacheOptions` holds the budgets and the tile size.
- **Test cost.** Every test runs on synthetic images of a few hundred pixels,
  in a temporary workspace whose `mipmap_options` sets a threshold of 64, with
  a cache whose tile size is 32 ([mipmaps.md § "The workspace setting"](../core/camera/mipmaps.md#the-workspace-setting)). No test reads or
  makes a large photograph.

## Interface

New, in `crates/sfmtool-core/src/camera/mipmap.rs` (with submodules for the
record, the region test and the tile decode), exported from
`sfmtool_core::camera`:

```rust
pub const DEFAULT_OUT_OF_CORE_LONG_SIDE: u32 = 1536; // what `sfm ws init` writes
pub const MAX_JPEG_SIDE: u32 = 65535; // a photograph with a larger side is refused
pub const PHOTOGRAPH_TILE_SIZE: u32 = 512;
pub const MIPMAP_RESTART_INTERVAL_MCUS: u16 = 16;
pub const MAX_REGION_INTERVAL_WIDTH: u32 = 1024;
pub const MIPMAP_DIR_NAME: &str = ".sfm-mipmaps";

/// A workspace's `mipmap_options` object. Every field is required in the
/// file; there are no defaults inside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MipmapOptions {
    pub out_of_core_long_side: u32,
    pub jpeg_quality: u8,                 // levels 1 and coarser
    pub full_resolution_jpeg_quality: u8, // `-mip0`
}

/// Every level's size for a photograph's size, and how many are stored.
/// `options` is `None` for a workspace without `mipmap_options`, or no
/// workspace: then every photograph is in core. A photograph with a side
/// over `MAX_JPEG_SIDE` has no layout: the cache refuses it first.
pub struct PyramidLayout { /* … */ }
impl PyramidLayout {
    pub fn for_size(width: u32, height: u32, options: Option<&MipmapOptions>, tile_size: u32) -> Self;
    pub fn levels(&self) -> usize;                 // full_levels
    pub fn level_size(&self, level: usize) -> (u32, u32);
    pub fn stored_levels(&self) -> usize;          // K, 0 when in core
    pub fn is_out_of_core(&self) -> bool;
    pub fn tiles(&self, level: usize) -> (u32, u32);
}

/// The region test of § "Which originals can be read by region": the layout
/// a region decode needs, or the first rule the bytes fail.
pub fn region_layout(bytes: &[u8]) -> Result<RegionLayout, NotRegionReadable>;

/// Where a photograph's mip level files are, or why it has none.
pub enum MipmapLocation {
    Files { root: PathBuf, relative: String, paths: Vec<PathBuf> },
    InMemory(NoFilesReason), // mipmap directory not writable
}
pub fn mipmap_location(photograph: &Path, layout: &PyramidLayout, level0_file: bool) -> MipmapLocation;

/// Make a photograph's mipmap from one decode, writing what can be written:
/// levels 1 to `K`, and `-mip0` when the original is not region-readable.
/// `decoded` is `ImageU8::read_rgb(photograph)` when the caller already holds
/// it (`build_sift_file` does), and is decoded here otherwise. Returns the
/// encoded bytes of every stored level, which the caller decodes rather than
/// reading the files back. Tells `on_level` each level as it is encoded, the
/// coarsest first.
pub fn build_mipmaps(
    photograph: &Path,
    decoded: Option<&ImageU8>,
    options: &MipmapOptions,
    tile_size: u32,
    progress: &Progress<'_>,
    on_level: &mut dyn FnMut(&MipLevelFile),
) -> Result<BuiltMipmap, MipmapError>;

/// One mip level file's bytes, record and restart index, read and checked once.
pub struct MipLevelFile { /* … */ }
impl MipLevelFile {
    pub fn open(path: &Path, expected: &ExpectedMipLevel) -> Result<Self, MipmapError>;
    pub fn from_bytes(bytes: Arc<[u8]>, expected: &ExpectedMipLevel) -> Result<Self, MipmapError>;
    pub fn record(&self) -> &MipLevelRecord;
    pub fn decode(&self) -> Result<ImageU8, MipmapError>;
    /// Pixels `x0..x1` by `y0..y1` of the level, equal to the same pixels of
    /// `decode`.
    pub fn decode_region(&self, x0: u32, y0: u32, x1: u32, y1: u32) -> Result<ImageU8, MipmapError>;
}

/// Region reads of a region-readable original.
pub struct OriginalRegions { /* … */ }
impl OriginalRegions {
    /// `Err` names the rule of the region test the file fails.
    pub fn from_bytes(bytes: Arc<[u8]>) -> Result<Self, NotRegionReadable>;
    /// Equal to the same pixels of `ImageU8::read_rgb(path)`.
    pub fn decode_region(&self, x0: u32, y0: u32, x1: u32, y1: u32) -> Result<ImageU8, image::ImageError>;
}

pub enum MipmapError { Io { path: PathBuf, error: std::io::Error }, Stale { path: PathBuf, why: String },
    Source(image::ImageError), Workspace(WorkspaceConfigError), TooLarge { width: u32, height: u32 },
    Encode(String), Cancelled }
```

In `sfmtool_core::workspace` (a new `mipmap_options` module beside
`feature_options`), with `WorkspaceConfigError` new beside it:

```rust
/// The `mipmap_options` object of `<root>/.sfm-workspace.json`: `None` when
/// the file has none, and an error naming the file, the key and the value for
/// a missing key, an unknown key or a bad value.
pub fn read_mipmap_options(root: &Path) -> Result<Option<MipmapOptions>, WorkspaceConfigError>;
```

Changed, in [photograph_cache.rs](../../crates/sfmtool-core/src/camera/photograph_cache.rs):

```rust
pub struct PhotographCacheOptions {
    pub budget_bytes: u64,          // default_budget_bytes(): tiles, file bytes, coarsest levels
    pub build_budget_bytes: u64,    // default_build_budget_bytes(): whole decodes in progress on this cache
    pub tile_size: u32,             // PHOTOGRAPH_TILE_SIZE; only tests change it
}

/// Gives each granted build's bytes back to its cache's count as it ends,
/// the rest on drop. Made by `PhotographCache::reserve_builds` and
/// `try_reserve_build`.
pub struct BuildReservation<'a> { /* … */ }
impl BuildReservation<'_> {
    pub fn granted(&self) -> usize;
    pub fn release_one(&self);
}
/// The default build budget, by the rule of `default_budget_bytes`: a
/// quarter of physical memory clamped to 1 to 16 GiB, or 4 GiB where it
/// cannot be read, overridden by `SFM_EXPLORER_BUILD_BUDGET_MB`
/// (`BUILD_BUDGET_ENV_VAR`). It keeps no state.
pub fn default_build_budget_bytes() -> u64;

/// What a kernel reads a photograph through, for every photograph: the
/// cache's handle to its tiles, or tiles the source owns.
pub struct TiledPhotograph { /* … */ }
impl TiledPhotograph {
    /// Every tile of the image's pyramid of `full_levels` depth, built by
    /// `ImageU8Pyramid::from_image` as the bindings built it before, so the
    /// same pixels; held by the source, outside any cache: the Python
    /// bindings given arrays, and tests (built in step 13e2).
    pub fn from_image(image: ImageU8) -> Self;
    /// A whole pyramid the cache handed out, read in place: a tile or region
    /// read copies only the pixels it asks for from the pyramid's levels, and
    /// the source keeps no tile, so a reader that moves holds no more than
    /// the pyramid it held before. Made in step 7, read by moved readers from
    /// step 9, and deleted in step 13e3 (built), when the cache keeps tiles
    /// and hands out tiled photographs made of them.
    pub fn from_pyramid(pyramid: Arc<ImageU8Pyramid>) -> Self;
    /// Every level's size and the number of levels, read with no pixel: what
    /// a reader that today reads `level(0)` or `num_levels()` only for the
    /// photograph's size reads instead.
    pub fn layout(&self) -> &PyramidLayout;
    /// The channels of every level's pixels, which a reader reads today as
    /// `level(0).channels()`.
    pub fn channels(&self) -> u32;
}

impl PhotographCache {
    pub fn new(options: PhotographCacheOptions) -> Self; // no `levels`: depth is by size
    /// Makes the mipmap of an out-of-core photograph whose files are missing or
    /// stale, in the path's slot on the calling thread, reserving that one
    /// build with `reserve_builds`, or waits for the caller already making it;
    /// reports through `progress`. Decodes an in-core photograph none of
    /// whose tiles are cached into tiles, kept as unused.
    pub fn get(&self, path: &Path, progress: &Progress<'_>) -> Result<Arc<TiledPhotograph>, GetError>;
    /// Reserve the estimated bytes of up to `wanted` builds against this
    /// cache's build budget with one compare-exchange loop on its count, at
    /// the moment the caller decides; never waits. Grants as many as fit, and
    /// one when none fit, so a kernel caller always makes progress.
    pub fn reserve_builds(&self, bytes_per_build: u64, wanted: usize) -> BuildReservation<'_>;
    /// A panel's reservation for one decode: `None` when it does not fit
    /// under the build budget, never one past it; the panel asks again on a
    /// later frame. Never waits.
    pub fn try_reserve_build(&self, bytes: u64) -> Option<BuildReservation<'_>>;
    /// A panel's decode of an in-core photograph into tiles, the `keys` it
    /// asked for kept as used and the rest as unused: reserves with
    /// `try_reserve_build`, then claims the slot only when no other caller is
    /// decoding in it, giving the reservation back if one is, and answers
    /// `Busy` without waiting when the reservation is refused or another
    /// caller holds the slot.
    pub fn try_decode_tiles(&self, path: &Path, keys: &[TileKey], progress: &Progress<'_>)
        -> Result<TryDecode, TileError>;
    /// The header, the tier, the level sizes and, out of core, the region
    /// answer; makes no mipmap. The blocking form of `layout`.
    pub fn read_layout(&self, path: &Path) -> Result<PhotographLayout, GetError>;
    /// Pixels `rect` of `level` as every kernel samples them, making the mipmap
    /// first if its files are missing or stale (`get`), then assembling it
    /// from tiles, in core or out. A level or
    /// rectangle outside the photograph, or an empty rectangle, is
    /// `OutOfRange`. Behind the Python class's `read_level`, `read_tile` and
    /// `read_region`.
    pub fn read_region(&self, path: &Path, level: usize, rect: [u32; 4], progress: &Progress<'_>)
        -> Result<ImageU8, PhotographReadError>;
    /// Makes its missing mipmaps, and decodes into tiles its in-core
    /// photographs none of whose tiles are cached, in parallel on the rayon
    /// pool, in groups sized by the build budget; the error type is today's.
    pub fn get_many(&self, paths: &[&Path], progress: &Progress<'_>)
        -> Result<(Vec<Option<Arc<TiledPhotograph>>>, GetManyTally), Cancelled>;
    // peek and peek_state answer `Arc<TiledPhotograph>` too; insert takes an
    // `ImageU8` and cuts it into the path's tiles; forget and clear as today;
    // `stats` counts the tile and file-byte entries, bytes, budget, hits and
    // misses, the whole decodes of in-core photographs, the build budget, the
    // bytes reserved for whole decodes, and the mipmaps made by this cache

    pub fn layout(&self, path: &Path) -> PeekedLayout;
    pub fn request_layout(&self, path: &Path);   // a header read, the region test, a record check
    pub fn peek_tile(&self, path: &Path, key: TileKey) -> PeekedTile;
    /// Only for a photograph whose layout is `Known`; never makes a mipmap; for
    /// an in-core photograph a miss decodes it into tiles, reserving with
    /// `reserve_builds` and waiting on the slot (a kernel's read; panels use
    /// `try_decode_tiles`). Answers `Changed` rather than waiting when its
    /// files have gone stale.
    /// The four calls that can decode a whole photograph take the caller's
    /// `&Progress`, polled for cancellation as `get` polls it.
    pub fn get_tile(&self, path: &Path, key: TileKey, progress: &Progress<'_>)
        -> Result<Arc<PhotographTile>, TileError>;
    /// A view-region pyramid of a photograph, assembled from its tiles
    /// ([view-region-pyramid.md](view-region-pyramid.md)).
    pub fn view_region_pyramid(&self, path: &Path, regions: &LevelRegions,
        progress: &Progress<'_>) -> Result<ImageU8Pyramid, TileError>;
    /// One level whole, for the 3D background, assembled from tiles.
    pub fn get_level(&self, path: &Path, level: usize, progress: &Progress<'_>)
        -> Result<Arc<ImageU8>, TileError>;
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

// `PeekedLayout` and `PeekedTile` as in § "The caches in memory".
pub enum TileError { Changed { path: PathBuf }, Unreadable { path: PathBuf, why: GetError },
    /// A level, tile or rectangle the photograph does not have.
    OutOfRange { path: PathBuf, why: String },
    /// The caller's `Progress` was cancelled during a decode; never remembered.
    Cancelled }
/// `Decoded` carries the tiles asked for, which under a budget of 0 are the
/// only copy.
pub enum TryDecode { Decoded(Vec<Arc<PhotographTile>>), Busy }
pub enum GetError { NotFound(PathBuf), Unreadable(String), Workspace(WorkspaceConfigError),
    /// A side of 65536 px or more, read from the header; in every workspace.
    TooLarge { path: PathBuf, width: u32, height: u32 }, Cancelled }

/// `read_layout`'s answer: the layout, and `None` in core or whether the
/// original is region-readable out of core.
pub struct PhotographLayout { pub layout: Arc<PyramidLayout>, pub region_readable: Option<bool> }
```

`ProjectedImage`'s `pyramid` field becomes a `&TiledPhotograph`. The
`SFM_EXPLORER_TILE_CACHE_MB` variable does not exist: the cache has one
budget, set by `SFM_EXPLORER_PHOTOGRAPH_CACHE_MB` as today.

### Errors from reading pixels

Rendering cannot fail today: `render_tile` returns an `ImageU8` and
`render_tile_with_grad_into` returns nothing
([sampler.rs](../../crates/sfmtool-core/src/camera/sampler.rs)). A render of a
tiled source can: its files can go stale or change under it, and a kernel
can be cancelled while it waits for tiles. So there is one error type for
everything a kernel can meet while reading photographs, in
`sfmtool_core::camera`:

```rust
/// Why a kernel could not read the photographs it was given.
pub enum PhotographReadError {
    Cancelled,
    /// A file's record differs from the one this kernel call read, or it went
    /// stale after the call's read phase found it current.
    Changed { path: PathBuf },
    /// The photograph does not exist (`GetError::NotFound`).
    NotFound { path: PathBuf },
    /// A side of 65536 px or more (`GetError::TooLarge`).
    TooLarge { path: PathBuf, width: u32, height: u32 },
    Unreadable { path: PathBuf, why: String },
    /// `read_region` was asked for a level or rectangle the photograph does
    /// not have, or an empty one.
    OutOfRange { path: PathBuf, why: String },
}
impl From<Cancelled> for PhotographReadError { /* … */ }
impl From<TileError> for PhotographReadError { /* … */ }
impl From<GetError> for PhotographReadError { /* … */ }
```

**The samplers stay infallible.** `render_tile`, `render_tile_with_grad_into`,
`render_tiles` and the `remap_*` functions keep their signatures and take
`&ImageU8Pyramid`, the region pyramid a render has already assembled, which
cannot fail; the tests of the samplers call them. Beside them, the **render
entry points**, which are the only way a kernel reads pixels (the companion's
§ "Where the pixels come from" says what each takes and returns):
`render_source_tile` and `render_source_tile_with_grad_into` for warp-map
renders, `render_source_affine` for the view selection's affine fast path,
`sample_source_grid` for the cluster refinement's grids,
`read_source_samples` for the grown plane's value and Sobel reads,
`read_source_rect` for the cluster refinement's `LevelTile` and the grey
reads of *Find Nearby Tracks*, and `clipped_share`. Each takes the source
(a `&ProjectedImage`, and so a `&TiledPhotograph`), the region pool the
kernel's caller made for this kernel call (`RegionPool::for_kernel_call()`,
whatever way the caller fetched its sources; a kernel that calls another
passes its own pool down) and the caller's `&Progress`, which a read that
decodes polls for cancellation (the parameter built in step 13e4a, and the
decode that polls it in 13e4b, built), and returns
`Result<_, PhotographReadError>`: it works out the regions the read needs
and copies them from the pool or reads through tiles, either of which can
fail for a photograph the cache reads, and never fails for one built
`from_image`. No kernel computes a region or touches a photograph tile.

**Kernels pass the error up.** Every kernel that reads from a
`ProjectedImage` calls the render entry points and returns the error
with `?` through its loops, including its parallel ones (a `try_for_each` or
a `collect::<Result<…>>` in place of a `for_each` or a plain `collect`). A
kernel that already returns `Result` with an error enum of its own gains a
`Photograph(PhotographReadError)` variant; one that returns plain values now
returns `Result<_, PhotographReadError>`. The viewer's and the bindings'
callers turn it into the step's failure (*The mipmap of … changed while
… ran; run it again*) or a Python `RuntimeError` naming the path. The changes,
each in the step that moves its readers:

| Step | Signatures that change |
|------|------------------------|
| 9 | Track View's and the Track Table's renders, which draw a cell's error state instead of the tile (built) |
| 10 | `ViewSources::fetch` in [state/edits.rs](../../crates/sfm-explorer/src/state/edits.rs), which returns `PhotographReadError` instead of `Cancelled` (`get_many` still returns `Cancelled`, which converts), and the bench kernels (`bench::evaluate`, `evaluate_rendering_bitmap`, `score_bitmap`, `fit`, `render_bitmap_in_place`, `search_geometry`, `set_stage`, `build_track_at_pixel`, `classify`) and the patch kernels they and the reference picks share (`view_selection::select_patch_views`, `select_patch_cloud_views`, `reference_view::render_view_tile`, `read_track`, `stored_bitmap::render_reference`, `keypoint_localize`'s and `keypoint_subpixel`'s entry points, `member_coherence`'s, `observation_reading::read_cloud_observations`, `spawn::spawn_candidate_tracks`), and every kernel that calls one of them or that the bench runs, since a kernel passes its caller's pool down and its errors up: the normal kernels the bench's normal steps run (`pairwise_flow_normal`, `grown_plane_normal`, `grid_plane_normal`, `finite_difference_normal`) and `refine_normals`, which runs them; the cluster refinement's entry points (`refine_cluster_patches`, `refine_cluster_patches_borrowed`), which the bench's evaluation of a cluster-stage track runs, and the bench's member-grid reads, which move to `sample_source_member_grid`, the source form of `sample_member_grid` that step 9 added beside it; `stored_bitmap::render_patch_bitmap`, `render_patch_cloud_bitmaps` and `refit_points`. Their callers that step 11 moves keep their fetch: each wraps the whole pyramids it holds with `from_pyramid` (the cluster-patches build, until 11b, its decoded images with `from_image`), makes the region pool for its call, and reports the error as its step's failure |
| 11 | the public `render_patch_bitmap_column` / `render_display_patch_bitmaps` in [display_bitmaps.rs](../../crates/sfmtool-core/src/patch/display_bitmaps.rs), which return `PhotographReadError` instead of `Cancelled`, and `WebExportError`, which gains `Photograph` (built, 11a); the viewer's bulk jobs (the conversion to embedded patches, *Refine all normals*, *Refit all points*, the cluster-patches build) and `sfm web-export` take the tiled photographs `get_many` answers since step 13e1, having wrapped its whole pyramids with `from_pyramid` before it, and the cluster-patches build fetches through the cache from 11b (built) |
| 12 | `read_patch_along_ray` and `depth_layers`, which take the caller's `&RegionPool` in place of the `&GreyImages` cache, and `far_field_sweep` and `find_nearby_tracks`, which took the pool in 10a and drop the cache; `PatchReadError` and `DepthLayerError` gain `Photograph(PhotographReadError)` in place of `GreyMismatch`, and `GreyImages`, with its `get` and `GreyImageError::NoSuchImage`, is gone (built). The plan also listed `guided_matches`, `nearby_cluster_tracks`, `constellation_seeds`, `nearby_points` and `distance_range`, which read the reconstruction, its index files and the cameras but no photograph, so their signatures do not change |
| 13 | `PhotographCache::get`, which takes the caller's `&Progress`, and `GetError`, which gains `Cancelled` (built, 13a); `PhotographCache::new`, which takes `PhotographCacheOptions` (built, 13b); the layout and tile requests, `TileError` and `TryDecode`, the four that can decode a whole photograph taking the caller's `&Progress` (built, 13c1); `display_photograph`, which gives way to `display_layout`, `display_tiles` and `display_level` (built for Image Detail, 13c2, the 3D background, 13c3, Track View, 13d1, and the Track Table and Measurement View's endpoint crops, 13d2, which deletes it); `get`, `get_many`, `peek` and `peek_state`, which answer `Arc<TiledPhotograph>`, `insert`, which takes an `ImageU8`, `PhotographReadError`, which gains `Changed`, `NotFound`, `TooLarge` and `Unreadable` with the conversions from `TileError` and `GetError`, and `from_pyramid`, which is deleted (13e: the answers' type built in 13e1, `from_image` holding its tiles in 13e2, `insert` taking an `ImageU8`, `keeps_every_tile`, which the image menu asks in place of `peek`, and `from_pyramid` deleted in 13e3, built, the errors in 13e4b, built); every render entry point, and the public kernels `refine_cluster_patches`, `refine_cluster_patches_borrowed`, `depth_layers`, `read_patch_along_ray`, `refine_view_against_reference`, `render_bitmap_in_place` and `spawn_candidate_tracks`, taking the caller's `&Progress`, with `Progress::silenced` new in `sfmtool-progress` (built, 13e4a) |
| 16 | `ImagePyramidSet.read`, the first binding whose photographs come from the cache. The bindings over the kernels above change with their kernels in steps 10 to 12, build their tiled photographs from arrays with `from_image`, and raise the error from then on, though tiles cut from arrays cannot fail |

The viewer's side, in
[photograph_requests.rs](../../crates/sfm-explorer/src/state/photograph_requests.rs):
`display_photograph` gives way to `display_layout` and `display_tiles`, which
act on the layout state as § "Readers on the GUI thread" says, queueing a photograph that is
`Building` and, for a `Known` photograph, starting one `get_tile` per missing
tile out of core and one `try_decode_tiles` per in-core photograph with the
tiles it is missing, at most one in flight per photograph, for every
photograph.
The *Build mipmaps* quiet task is `state/build_mipmaps.rs`: a job that takes
one chunk of photographs from its source's shared list, in the order § "Who
makes the mipmaps, and when" gives, calls `get_many` on it and ends, its
source queueing it again while the list is not empty.

## Python bindings

The user has decided that Python gets the same machinery as the viewer. All
of it is in `sfmtool-core`; the bindings are thin wrappers over it, with no
logic in Python ([STANDARDS.md](../STANDARDS.md#logic-lives-in-rust-not-in-python-wrappers)).
**Only these parts are the viewer's own**: drawing photograph tiles as GPU
textures, the panels queueing photographs for the *Build mipmaps* quiet task,
and the task itself. Everything else in
this design (the tiers, the mipmaps, the slots and build reservations, tiles,
levels, regions, view-region pyramids, and the cache's statistics) is core and
reachable from Python. Core keeps no global state: every count, slot and entry
belongs to a `PhotographCache` instance, and the viewer's session-wide cache
is the one it holds in `AppState`, as a script holds its own.

**A new submodule, `sfmtool.photographs`** (`_sfmtool.photographs`,
`crates/sfmtool-py/src/photographs.rs`), the fourteenth, named for what the
photograph cache holds ([GLOSSARY.md](../GLOSSARY.md), *photograph cache*).
It is wired as `sfmtool.workspace` was: an `install_submodule` call in
`lib.rs` and its name in that file's list of submodules, its public module
`src/sfmtool/photographs.py` re-exporting its `__all__`, its row in the index
of [python-bindings.md](../python-bindings.md), its name in
`_LAZY_SUBPACKAGES` and the `if TYPE_CHECKING:` imports of
`sfmtool/__init__.py` and in `tests/test_lazy_loading.py`'s list, so
`import sfmtool` loads nothing new, and a `tests/rust_bindings/photographs/`
package with `test_photographs_registration.py`, which
`tests/test_module_layout.py` already allows to name the extension. It
is not `sfmtool.fileio`, whose names are the I/O of sfmtool's file formats,
which mip level files are not ([mipmaps.md § "What is written in one"](../core/camera/mipmaps.md#what-is-written-in-one)), nor `sfmtool.patches`,
whose names are the patch kernels'.

```python
class PhotographCache:
    def __init__(self, budget_bytes: int | None = None,
                 build_budget_bytes: int | None = None): ...
    def layout(self, path) -> PhotographLayout: ...
    def read_level(self, path, level: int, *, progress=None) -> np.ndarray: ...
    def read_tile(self, path, level: int, col: int, row: int, *, progress=None) -> np.ndarray: ...
    def read_region(self, path, level: int, x0: int, y0: int, x1: int, y1: int,
                    *, progress=None) -> np.ndarray: ...
    def build_mipmaps(self, paths, *, progress=None) -> dict: ...
    def stats(self) -> dict: ...
```

- **`PhotographCache(budget_bytes=None, build_budget_bytes=None)`** holds
  one core `PhotographCache`; an omitted budget is the core default
  (`default_budget_bytes()` for the tiles and file bytes,
  `default_build_budget_bytes()` for whole decodes in progress), and a budget
  of 0 keeps nothing, as in the core. Every call on one
  `PhotographCache` shares its build count, so a script that holds one cache
  gets the same bound on builds in progress as the viewer (§ "Building a mipmap
  in the slot"). The tile size is not an argument: it is a
  constant ([mipmaps.md § "The workspace setting"](../core/camera/mipmaps.md#the-workspace-setting)).
- **`layout(path)`** returns a frozen `PhotographLayout` with `width`,
  `height`, `out_of_core`, `levels`, `level_sizes` (a list of `(width,
  height)` per level), `stored_levels` (`K`, 0 in core) and
  `region_readable` (`None` in core). It reads the header, the workspace's
  `mipmap_options` and, out of core, the region answer from a current record
  or the region test; it makes no mipmap. It is core's `read_layout`, the
  blocking form of `layout` and `request_layout`, and its
  `region_readable` is that call's `PhotographLayout::region_readable`.
- **`read_level`, `read_tile` and `read_region`** return a level, a photograph
  tile's core (without its border), or the pixels `x0..x1` by `y0..y1` of a
  level, as `(H, W, 3)` `uint8` arrays in the `y_x_rgb` layout of
  `read_image_rgb`, C-contiguous and **read-only** (numpy's writeable flag
  cleared, as the reconstruction's array getters are; a caller that edits
  takes `.copy()`). Their values are the pixels every kernel samples for that
  photograph: in core the exact levels of its in-memory pyramid; out of core
  level 0 from the original or `-mip0`, levels 1 to `K` from the mip level files,
  and the levels below `K` from level `K`. Out of core, a read makes the
  photograph's mipmap first if its files are missing or stale, in its slot, as
  any `get` does, so each takes `progress=`; a read of a current file decodes
  only the tiles it needs. Each is a call to core's `read_region` (a level is
  its whole region; a tile is its key's rectangle) with the GIL released.
- **`build_mipmaps(paths, progress=None)`** makes the mipmaps of the
  out-of-core photographs among `paths`, photographs' paths (`sfm ws
  mipmaps` expands directories itself with `expand_paths`, as `sfm sift`
  does; step 15e's note says why) through the cache's
  `get_many`, so a mipmap already made in this cache, or by another cache or
  process, is
  read rather than made again, and the build budget and its reservations
  apply. It returns a dict, as the bindings' other reports are dicts
  (`write_web_export`'s, `pick_images`' through `serde_to_py`, the bench's),
  with the keys `made`, `current` and `in_core` (lists of the paths in each
  case, as given) and `failed` (a list of `(path, message)` pairs). **It is the one way to
  build mipmaps from Python**; there is no free function for it. `sfm ws mipmaps` makes a
  `PhotographCache()` and calls it; the Python SIFT extraction calls it once
  its `.sift` files are written, on one cache the extraction run holds (§ "Who
  makes the mipmaps, and when").
- **`stats()`** returns a dict, as reports are, with the keys `entries`,
  `bytes`, `budget_bytes`, `hits` and `misses` (the tiles and file bytes the
  cache holds), `decodes` (whole decodes of in-core photographs, counting a
  photograph decoded again after its tiles were evicted),
  `build_budget_bytes` and `build_reserved_bytes` (the cache's count of whole
  decodes in progress), and `mipmaps_made`. `PhotographLayout` stays a frozen
  class: it is a value describing one photograph that a caller passes around
  and reads fields of, as `ImagePyramidSet` is, where a report or a set of
  statistics is a snapshot printed or logged once.
- **Errors raise, never panic.** A missing file raises `FileNotFoundError`
  and an unreadable one `OSError`, as `read_image_rgb` does; a level, tile or
  rectangle outside the photograph or an empty rectangle
  (`PhotographReadError::OutOfRange`), and a negative index or budget, raise
  `ValueError`, and an argument that is not an integer raises `TypeError`
  from PyO3's conversion; an invalid
  `mipmap_options` raises `RuntimeError` naming the key, as
  `load_workspace_config` does; a file whose record changes during a read
  raises `RuntimeError` naming the path (`PhotographReadError::Changed`); a
  photograph with a side of 65536 px or more raises `OSError` naming it and
  its size (`GetError::TooLarge`).
  `progress` is a `ProgressCounter`, as in every other binding, and counts
  mipmaps made; Ctrl+C stops a read or a build within one stage through the
  binding's `SignalWatch` and raises `KeyboardInterrupt`, as `pick_images`
  does. Each message names the path.

**An optional `cache=` argument on every binding that reads a
reconstruction's photographs by path**: `ImagePyramidSet.read(recon_or_paths,
workspace_dir=None, cache=None, progress=None)` and
`sfmtool.fileio.write_web_export(…, cache=None)`. `read`'s first argument is
an `SfmrReconstruction`, whose `image_names` it reads under `workspace_dir`
(the reconstruction's own workspace when omitted), or a list of photograph
paths, read in that order with no check of their sizes at the read (a kernel
checks the set against its scene); a `CameraViews` is refused, since it names
no photographs (open question 20). The kernels take arrays or
an `ImagePyramidSet`, not paths. The other bindings that read photographs by
path decode each whole and keep no pyramid, so they take no cache:
`read_image_rgb`, `read_image_rgba` and `image_has_alpha` (row P1), and the
SIFT extraction of `build_sift_files`, `pick_images` and the `.sift`
conversion (row 21). **There is no
implicit process-wide default cache.** When `cache` is omitted the call makes
its own short-lived cache, of the default budget, and drops it when
it returns; nothing is kept between calls unless the caller keeps it. A script
that wants reuse across calls holds a cache itself and passes it:

```python
from sfmtool.patches import ImagePyramidSet
from sfmtool.photographs import PhotographCache

_photographs = PhotographCache()  # held by this script, reused by every call


def pyramids_for(recon):
    # The tiles and mipmaps this reads stay in `_photographs` for the next call.
    return ImagePyramidSet.read(recon, cache=_photographs)
```

`ImagePyramidSet.read` reads its photographs as tiled photographs from the
cache, read by view-region pyramids, so each is read the way the viewer reads
it, making missing mipmaps on the way, and the set holds no whole pyramid of
any photograph. The `images` argument of the constructor and of every kernel
keeps meaning the arrays' exact levels, cut into tiles the set owns
(`TiledPhotograph::from_image`).

`load_workspace_config` validates `mipmap_options` through a binding over
`read_mipmap_options`, in `sfmtool.workspace`.

## Implementation steps

Each is one pull request, useful on its own, with its tests, and each leaves
every reader working. Steps 1 to 3 end the crash with in-memory pyramids,
before any file is written, and step 4 bounds the one other whole-photograph
reader that runs unbounded. Steps 5 to 7 build the mip level files and the
view-region pyramids without changing any reader. Step 8 limits how
elongated a cluster member's shape may be, which bounds each request for the cluster
refinement's own copies of a level before it reads tiles. Steps 9 to 12 move the
readers to view-region pyramids, one group per pull request, while the cache
still hands out whole pyramids: a moved reader wraps the pyramid it gets in a
`TiledPhotograph` (`from_pyramid`), so readers that have moved and readers
that have not both work. Step 13 makes the cache hand out tiled photographs
only, once every reader takes them, in thirteen pull requests: the slot and the
build budget first, then the cache's tile requests and the panels, then the
tiles. Steps 14 to 16 build the out-of-core tier, step 14 in seven pull
requests, without turning it on anywhere but in tests: until step 17, neither
`sfm ws init` nor any other command writes `mipmap_options`, so no workspace
has the object unless its owner writes it by hand, and every photograph stays
in core. Step 17 turns the tier on. Every step is built.

1. **Depth by size.** Built: photograph pyramids of `full_levels` depth
   everywhere, `PhotographCache::new` without `levels`, and no
   `DISPLAY_PYRAMID_LEVELS`; specified in
   [core/camera/photograph-cache.md](../core/camera/photograph-cache.md).
2. **Image Detail draws photograph tiles.** Built: the panel cuts photograph
   tiles from the cache's in-memory pyramid itself, a few per frame, with the
   level rule, the coarser fallback, the per-frame upload cap of 16, the
   texture limit of 256 and the zoom limit `max(32, 8 / fit_scale)`, so no
   texture is larger than 514 x 514 and Image Detail no longer crashes on any
   photograph, though a 200-megapixel one is still read whole until step 13e;
   specified in [gui/multi-panel-image-browser.md](../gui/multi-panel-image-browser.md)
   § "Image Detail" and § "Image Detail: 2D pan and zoom navigation". Two
   parts of § "Image Detail" were not in it and came with the cache's
   tile requests in step 13c2 (built): asking for the next image's tiles a frame ahead while
   animation plays, and the `level` and `tiles_pending` fields of
   `get_image_detail_view`. The zoom limit's `fit_scale` is in panel points, the unit
   the panel's view arithmetic and `set_image_detail_view` work in.
3. **The 3D background** from one level by the viewport and the limit, off the
   GUI thread, through the cache, with a remembered failure. Built: the level
   comes from the cache's in-memory pyramid through `display_photograph`, which
   decodes on the rayon pool on a miss (on the GUI thread only under a cache
   budget of 0), and no texture exceeds the limit, which ends the crash in
   camera view; specified in
   [gui/camera-views.md](../gui/camera-views.md) § "Photograph background".
4. **Image readings at save, bounded.** Built: `gather_image_readings` takes
   the rows that need no decode in parallel, then decodes the photographs
   left with at most as many at once as fit in half of physical memory at 16
   bytes per pixel of the largest, by `concurrent_images_in_memory`, the rule
   `.sift` building is sized by; specified, with the measurement, in
   [formats/sfmr-file-format.md § Implementations](../formats/sfmr-file-format.md#implementations).
5. **Mip level files.** Built: image discovery skips `.sfm-mipmaps`
   (`expand_paths`, the rig-config converter, the `.camrig` pattern matcher
   and the named-image-list walk); `PyramidLayout`, the region test with its
   interval-width rule, `mipmap_location`, the record, `MipmapOptions` and
   `read_mipmap_options` with their validation, `build_mipmaps` with `-mip0`,
   `MipLevelFile::open` and `decode`, the staleness test (`check_mipmap` over
   a `MipmapSource`), the writers' rules, `CACHEDIR.TAG`, and the refusal of
   a side of 65536 px or more in `PhotographCache::get`, which now answers
   `Result<_, GetError>`; specified in
   [core/camera/mipmaps.md](../core/camera/mipmaps.md) with its tests. No
   reader uses the files yet. Three parts of the interface above took another
   shape or wait for a later step: `mipmap_location(photograph)` takes only
   the path and answers `Option<MipmapLocation>` (`None` outside every
   workspace), since whether the files can be written is found by writing
   them, which `build_mipmaps` reports in `BuiltMipmap::not_written`;
   `photograph_layout(path, tile_size)` gives a photograph's layout and
   location from its header and workspace, with no cache, and the cache's
   `read_layout` and `PhotographLayout` come in step 16, with the class they serve; and the hash answer is remembered by a `MipmapSource` value,
   whose keeping per cache and spelling is step 14's. Measured outside the tests: the region test passes all 17 of the user's
   photographs (§ "Which originals can be read by region"), and one build of
   a 200-megapixel photograph peaks at about 920 MiB and takes 1.8 s
   (§ "Measurements behind the files").
6. **Tile and region decodes.** Built: the restart index,
   `MipLevelFile::decode_region` and `OriginalRegions`, specified in
   [core/camera/mipmaps.md § "Reading a rectangle"](../core/camera/mipmaps.md#reading-a-rectangle)
   with its tests, and `build_mipmaps` reading the header before it decodes,
   so an in-core or too-large photograph is refused without a decode. The
   step also corrected `MipLevelFile::decode`, which read the
   `ColorTransform::None` samples as interleaved where `jpeg-decoder` returns
   each line's Y, Cb and Cr samples in turn. Three parts of the design above
   took another shape: an empty or out-of-range rectangle of a mip level file
   is the new `MipmapError::OutOfRange { path, why }`; `OriginalRegions` also
   gives the original's `width` and `height`; and a region decode's right
   margin is one MCU rather than at least 16 px, since a frame a whole number
   of MCUs wide converts only its last 8 px with the scalar formula, and a
   margin of 16 px could not be told from one MCU by any test.
   Measured outside the tests (§ "A tile of a mip level file", § "A region
   of level 0"): 2.9 to 4.7 ms a tile of level 1 and 0.9 to 2.6 ms a region
   of the original, against the 2.3 to 6 ms and 2.6 to 3.5 ms measured
   before.
7. **View-region pyramids** ([view-region-pyramid.md](view-region-pyramid.md)
   steps 1 and 2), split in two pull requests:
   - **7a, the companion's step 1. Built:** regions in `ImageU8Pyramid`
     (`PyramidLevel`, `from_regions`, `level_origin`) and every sampler, the
     tile-addressed path with its window of pinned tiles, the kernel call's
     region pool, the render entry points (`render_source_tile` and its
     gradient form, `render_source_affine`, `sample_source_grid`,
     `read_source_samples`, `read_source_rect`) with `required_regions` and
     `LevelRegions::for_samples` behind them, `TiledPhotograph` with
     `from_image`, `from_pyramid`, `layout` and `channels`, and
     `PhotographReadError`, exercised on photographs cut into tiles in memory;
     specified in
     [core/camera/view-region-pyramids.md](../core/camera/view-region-pyramids.md)
     with its tests. No reader changes. Three parts took another shape than
     the interface above: `PhotographReadError` has only the variants a
     photograph in memory can meet, `Cancelled` and `OutOfRange`, whose
     `path` is `None` for such a photograph, and gains the others, with the
     conversions from `TileError` and `GetError`, when the cache hands out
     tiled photographs (step 13e); `TileKey` and `PhotographTile` live in
     `camera::tiled_photograph` beside `TiledPhotograph`; and a sampler
     subtracts a region's origin from its integer tap addresses after the
     `floor`, not from the coordinate, so the fractions are the whole level's
     for every coordinate.
   - **7b, the companion's step 2. Built**, so step 7 is complete: the
     regions of a copied read assembled from a `TiledPhotograph`'s tiles with
     the reuse margin, kept for reuse across renders, and the clipped share
     read over tiles, exercised on photographs cut into tiles in memory;
     specified in
     [core/camera/view-region-pyramids.md](../core/camera/view-region-pyramids.md)
     with its tests. No reader changes. Two parts took another shape than
     the interface above: the entry points but `read_source_rect` take the
     view's `ViewRegions`, made over the source and the kernel call's pool
     and holding the regions kept for that view, in place of the source and
     the pool; and the clipped share over tiles is `source_clipped_share`,
     beside the whole-pyramid `clipped_share` that `render_view_tile` keeps
     calling until step 10.
8. **The cluster members' elongation limit.** Built: `MAX_PATCH_ELONGATION`,
   the refusal at validation, the evaluation scored out of frame past the
   limit, `ClusterRefineResult::elongated_members` and
   `elongated_evaluations`, the bench's `EvaluateReport::elongated`, the
   counts in the summaries of `sfm cluster-patches` and the viewer's
   cluster-patches build, and the spec amendments below, with the tests
   below; the format version did not change, since `not_evaluated` keeps its
   meaning ([formats/format-versions.md](../formats/format-versions.md#when-a-formats-version-changes)).
   Measured with `sfm cluster-patches`' defaults on copies of the three
   checked-in captures, from features extracted and clustered afresh: no
   detected shape is past the limit (sfmtool's SIFT shapes are
   similarities), so no member is refused at validation and no cluster's
   reference changes; the cascades scored 744, 2847 and 1260 evaluations out
   of frame for their elongation (seoul_bull_sculpture, dino_dog_toy,
   kerry_park), which changed the outputs of 35 of 10985, 167 of 137708 and
   71 of 39381 members and the status of 1, 12 and 3 of them, and left no
   measured shape past the limit where 21, 83 and 42 had been. The largest
   `LevelTile` one member held fell from 155,040 to 118,720 px on
   dino_dog_toy and stayed at 13,528 and 24,235 px on the other two. The
   design follows (the companion's § "The members' elongation limit"),
   amending
   [core/patch/cluster-patch-refinement.md](../core/patch/cluster-patch-refinement.md)
   (§ "Which member anchors, and which members are eligible", § "The warp
   and the cascade", and the constants under § "Parameters"), the
   `not_evaluated` status in
   [formats/matches-file-format.md](../formats/matches-file-format.md), the
   summary of [cli/image-feature/cluster-patches-command.md](../cli/image-feature/cluster-patches-command.md),
   and a [GLOSSARY.md](../GLOSSARY.md) row for a shape's elongation, the ratio
   of its larger singular value to its smaller, which also settles
   [gui/track-view.md](../gui/track-view.md): its *Self-sim.* cell calls
   `1 − (minor / major)²` an ellipse's elongation, and the step rewords that
   line to state the bar as a minor axis at most 0.71 of the major, which is
   an elongation of at least `√2` in the glossary's sense:
   `MAX_PATCH_ELONGATION`, the refusal at validation, the evaluation scored
   out of frame past it, and the counts in `ClusterRefineResult` and the
   summaries. It changes results whether or not tiles are used, so it is its
   own pull request, before the
   cluster refinement reads tiles in step 10. Tests: a member whose shape's
   elongation is exactly the limit is refined, and one just past it is
   `not_evaluated` and counted as elongated; a cascade driven towards a map
   past the limit retreats and returns a shape within it; on a cluster whose
   members and evaluations all stay within the limit, every result is
   bit-identical to the result on the same clusters with the limit switched
   off. Measurement, in the pull request: on copies of
   the checked-in reconstructions (`seoul_bull_sculpture`, `dino_dog_toy`,
   `kerry_park`) and on the 200-megapixel capture of § "The problem,
   measured" if it is at hand, `sfm cluster-patches` before and after: the
   members refused at validation, the evaluations scored out of frame for
   their elongation, the largest `LevelTile` one member held, the clusters
   whose reference changed, and the members whose status changed.
9. **Track View and the Track Table** on view-region pyramids (companion
   step 3), with the step-9 signature changes, peeking the cache's whole
   pyramids as today and reading them through `from_pyramid`; Track View
   peeks tiles since step 13d1, and the Track Table since step 13d2.
   **Built**, before step 8: each panel owns a region
   pool of `REGION_POOL_BYTES_PER_THREAD`; Track View's track-stage tiles
   and hover views render through `render_source_tile`, its cluster-stage
   ones through `sample_source_member_grid`, and the crops of both panels
   and their hover views read through `read_source_samples`, each with a
   `ViewRegions` dropped once its picture is cut; a read that fails draws
   the cell's error state, *unreadable* with why on hover in Track View and
   the *Crop* cell's *unreadable* in the Track Table; specified in
   [gui/track-view.md](../gui/track-view.md),
   [gui/track-table.md](../gui/track-table.md) and
   [core/camera/view-region-pyramids.md](../core/camera/view-region-pyramids.md)
   with its tests. One part took another shape than the companion's step 3:
   the cluster-stage tiles read through `sample_source_member_grid`, a source
   form of `sample_member_grid` added beside it, since the member grid's
   level rule is private to the cluster refinement and a panel cannot work
   out the grid's region; `sample_member_grid` keeps its signature.
10. **The bench's jobs and the reference picks** (companion step 4), with the
    step-10 signature changes, one region pool made by each bench step and each
    reference pick for its kernel call, and the bindings over these kernels
    building their tiled photographs from arrays with `from_image`. The
    kernels the bulk readers share with the bench, the normal kernels and the
    cluster refinement among them, change here, and the bulk readers wrap
    what they fetch until step 11.

    The step is three pull requests, 10a to 10c. The bench's kernels call
    one another through `evaluate_rendering_bitmap`, which every normal step
    and every fit ends with, and through `fit`, which the finite-difference
    normal, *Find Nearby Tracks*' far-field sweep and `refit_points` run;
    and nearly every one of them reads through `render_view_tile`, which the
    stored bitmap, the keypoint kernels, the observation reads and the
    reference picks share. A kernel that moves takes a pool from its callers
    and returns the error to them, so moving any one read moves every caller
    above it, and moving `render_view_tile` first would move the whole step
    at once. So the split runs from the top down: 10a gives the bench's
    kernels their pool and their error, with the reads in the bench's own
    code, and 10b and 10c each move the reads of the patch kernels beneath
    them, with the signatures those kernels' other callers see. Until its
    sub-step, a patch kernel reads the whole pyramid its tiled photograph
    holds, as it read the `ProjectedImage`'s pyramid before, so every result
    is identical after each of the three.

    - **10a, the plumbing and the bench's own reads.** **Built.**
      `ProjectedImage` carries a `&TiledPhotograph` in place of its pyramid,
      and every caller that builds one wraps the whole pyramids it holds with
      `from_pyramid` (`ViewSources`, the display bitmaps, the Track Table's
      reference picks, *Add image to tracks*, whose kernel takes tiled
      photographs) or, in the bindings, cuts the pyramids it builds from its
      arrays the same way, which are the pixels `from_image` builds. The bench
      kernels whose reads reach a moved read (`evaluate`,
      `evaluate_rendering_bitmap`, `fit`, `set_stage`, `build_track_at_pixel`,
      the four normal kernels, `refine_normals`, `refit_points`), and the
      kernels of step 12 that run `fit` or `set_stage` (`far_field_sweep`,
      `find_nearby_tracks`), take the caller's `&RegionPool` and gain a
      `Photograph(PhotographReadError)` error (`set_stage`'s arrives inside
      `StageError::Fit`). Where an inner step turns its failures into values
      (the grown plane's renders, the track-at-pixel cascade's members, the
      far-field sweep's refits, the points of `refine_normals` and
      `refit_points`), the first read that failed is kept in a `ReadFailure`
      and the kernel returns it in place of its result. Each bench step, *Refine
      all normals*, *Refit all points* and each binding make the pool for
      their call, and each of the viewer's steps writes a `read regions` phase
      whose note counts the region pyramids its reads assembled and reused and
      the reads that took the tile path ([gui/bench.md](../gui/bench.md)). The
      reads that moved are the bench's own: the grid-plane normal's context
      renders and the pairwise-flow crops through `render_source_tile`, the
      grown plane's value and Sobel reads through `read_source_samples`, and
      the cluster stage's member grids (the template and the self-similarity
      tiles) through `sample_source_member_grid`, specified in
      [core/bench/editable-track.md](../core/bench/editable-track.md) §
      "Each step reads through the region pool its caller made" and
      [core/camera/view-region-pyramids.md](../core/camera/view-region-pyramids.md).
      `classify` reads no pixel (`TrackRays` reads only cameras), so its
      signature stays. `score_bitmap` and `render_bitmap_in_place` read only
      through the reference view and the stored bitmap, so they move in 10c,
      and `search_geometry` only through the view selection, so it moves in
      10b. The bindings raise a photograph error as the `ValueError` they
      raise for every refusal of these kernels, its sentence naming the
      photograph; the `RuntimeError` of § "Errors from reading pixels" comes
      in 10c, for the bindings over every kernel of the step at once, before
      step 16 gives a binding photographs that can fail to read.
    - **10b, the cluster refinement and the view selection.** **Built.**
      `refine_cluster_patches` and `refine_cluster_patches_borrowed` take
      tiled photographs and the pool and return
      `Result<ClusterRefineResult, PhotographReadError>`: the grid samplers
      (the member gate's grid, the template, the grid at the refined shape and
      the piecewise stage's working patches) read through
      `sample_source_grid` with one `ViewRegions` per member, dropped when its
      cluster is done, and `LevelTile` through `read_source_rect`, the tile
      keeping the `SourceRect` so its bytes stay reserved while it lives; the
      tile's rectangle is worked out as before from the maps and the level's
      size, now read from the layout, so its corner and mean are unchanged.
      `select_patch_views` and `select_patch_cloud_views` take the pool and
      return `PhotographReadError` in place of `Cancelled`, with the affine
      fast path through `render_source_affine` and the full render through
      `render_source_tile`, by `normalized_source_stack`, a source form of
      `normalized_stack` added beside it, since member coherence shares
      `normalized_stack` and moves in 10c; a track view's reference render
      and its own score share one `ViewRegions`. `search_geometry` takes the
      pool and gains `GeometrySearchError::Photograph`. Their callers make
      the pool or pass theirs down: the bench's cluster-stage evaluation,
      *Create track at pixel*'s geometry search, which keeps a failed read in
      its `ReadFailure`, the viewer's geometry search, which writes a `read
      regions` note ([gui/bench.md](../gui/bench.md)), the viewer's
      cluster-patches build, which builds its decoded images with
      `from_image` and makes a pool for each batch of clusters, and the
      bindings, which raise a photograph error as `ValueError`. Specified in
      [core/patch/cluster-patch-refinement.md](../core/patch/cluster-patch-refinement.md),
      [core/patch/patch-view-selection.md](../core/patch/patch-view-selection.md)
      and [core/camera/view-region-pyramids.md](../core/camera/view-region-pyramids.md),
      with tests in
      [patch/region_read_tests.rs](../../crates/sfmtool-core/src/patch/region_read_tests.rs)
      and [bench/tests/region_reads.rs](../../crates/sfmtool-core/src/bench/tests/region_reads.rs).
      The plan named two callers that are not: the grid-plane normal's cell
      registration calls neither kernel, and its renders moved in 10a, and
      `track_views` reads no pixel and only lists the track views a caller
      hands the view selection. It is a pull request of its own because the
      cluster refinement's `LevelTile` copies are the subtlest reads of the
      step.
    - **10c, the reference view and everything that reads through it.**
      **Built**, so step 10 is complete. `render_view_tile` renders through
      `render_source_tile` and reads its clipped share through
      `source_clipped_share`, through a `ViewRegions` its caller passes;
      `read_track`, `render_reference` and member coherence's
      `member_zncc_matrix_reporting` take one per view, so member
      coherence's renders reuse what the tiles read, by
      `normalized_source_stack`, and `normalized_stack` is deleted.
      `member_zncc_matrix`, `validate_member_coherence`,
      `validate_patch_cloud_member_coherence`, `render_patch_bitmap`,
      `render_patch_cloud_bitmaps`, `read_cloud_observations`, the
      localizer's and the sub-pixel refiner's entry points (their renders,
      the reference-view rule they resolve their reference by, their
      blur-matched scores and the stored bitmap), `spawn_candidate_tracks`,
      `score_bitmap` and `render_bitmap_in_place` take the pool and return
      the error (`LocalizeError`, `CloudRenderError` and `SpawnError` gain a
      `Photograph` variant; the batches return `PhotographReadError` in place
      of `Cancelled`), each making one `ViewRegions` per view of the patch in
      hand. The display bitmaps' kernels take the pool and keep returning
      `Cancelled` until step 11, since every photograph they read is a whole
      pyramid. The Track Table's reference picks make one pool for each
      `render_reference` call, and a read that fails gives no pick. The read
      of a tiled photograph's whole pyramid (`held_pyramid`) is left only to
      the grey reads of *Find Nearby Tracks*, which step 12 moves, and to the
      tests that compare a read with the whole pyramid's. The bindings over
      every kernel moved in 10a to 10c raise a photograph error as a
      `RuntimeError` carrying the kernel's sentence, which names the
      photograph, or says it is one in memory and carries the read's message,
      in place of the interim `ValueError`. Specified in
      [core/patch/reference-view.md](../core/patch/reference-view.md),
      [core/patch/member-coherence-validation.md](../core/patch/member-coherence-validation.md),
      [core/patch/patch-keypoint-localization.md](../core/patch/patch-keypoint-localization.md),
      [core/patch/keypoint-subpixel-refinement.md](../core/patch/keypoint-subpixel-refinement.md),
      [core/patch/candidate-track-spawning.md](../core/patch/candidate-track-spawning.md),
      [core/reconstruction/add-image-to-tracks.md](../core/reconstruction/add-image-to-tracks.md),
      [core/bench/editable-track.md](../core/bench/editable-track.md) and
      [core/camera/view-region-pyramids.md](../core/camera/view-region-pyramids.md),
      with tests in
      [patch/region_read_tests.rs](../../crates/sfmtool-core/src/patch/region_read_tests.rs)
      and [bench/tests/region_reads.rs](../../crates/sfmtool-core/src/bench/tests/region_reads.rs).
      The plan left out three readers through the reference view that move
      here with it: *Add image to tracks* (`add_image_to_tracks`, which takes
      the pool and gains `AddImageToTracksError::Photograph`, and the
      `TrackReferences` it measures each point with, whose `build`, `search`
      and `score` take the views' regions and return a read that fails rather
      than a refusal); `refine_view_against_reference`, its sub-pixel step;
      and the grid-plane and grown-plane normals, whose reference-view rule
      reads through regions made from their pool.
11. **Bulk kernels**: the display patch bitmaps, the conversion to embedded
    patches, the web export, *Refine all normals*, *Refit all points* and the
    cluster-patches build (companion step 5) fetch tiled photographs, with
    the step-11 signature changes and their bindings.

    Only the cache can hand out a tiled photograph, and it handed out whole
    pyramids until step 13e1, so what these callers fetch changed there:
    until then each wrapped the whole pyramids `get_many` answered with
    `from_pyramid`, as the bench's `ViewSources` did, and step 13e1's
    `get_many` answers `Arc<TiledPhotograph>` to all of them at once. The
    companion's step-5 tests that count decodes, and its test of a budget
    smaller than one photograph's level 0, need the tiled cache, so they come
    with step 13e5. Before it, the step is three pull requests, 11a to 11c:

    - **11a, the display bitmaps' error.** **Built.**
      `render_patch_bitmap_column` and `render_display_patch_bitmaps` return
      `PhotographReadError` in place of `Cancelled`, and 10c's interim rule,
      which left out a cached photograph whose pyramid was not whole, is gone:
      a photograph that cannot be decoded, or is not the size its camera says
      (now read from its layout), is still left out of every patch's views,
      and the first read of a photograph handed to the render that fails is
      the call's error. Their callers make it the step's failure: the open
      fails the file (*Failed to render the patch bitmaps of …: a photograph
      could not be read: …*), the conversion to embedded patches is refused
      with the same clause, and `sfm web-export` returns
      `WebExportError::Photograph`, which `write_web_export` raises as a
      `RuntimeError`. After the render, a `read regions` stage notes the
      pool's counts, as the bench's steps, *Refine all normals* and *Refit
      all points* do, so the open and the conversion write it too. The reads
      are those 10c moved, so the bitmaps are identical bit for bit; their
      read paths are tested in
      [patch/region_read_tests.rs](../../crates/sfmtool-core/src/patch/region_read_tests.rs).
      Specified in [gui/background-tasks.md](../gui/background-tasks.md),
      [core/camera/photograph-cache.md](../core/camera/photograph-cache.md)
      and [cli/visualization/web-export-command.md](../cli/visualization/web-export-command.md),
      with tests in
      [display_bitmaps/tests.rs](../../crates/sfmtool-core/src/patch/display_bitmaps/tests.rs),
      [state/open/tests.rs](../../crates/sfm-explorer/src/state/open/tests.rs)
      and [state/edits/tests.rs](../../crates/sfm-explorer/src/state/edits/tests.rs).
    - **11b, the cluster-patches build through the cache.** **Built.** The
      viewer's build reads its photographs with `get_many` through the
      viewer's cache (row 19 of § "Every reader of photograph pixels"),
      wrapping each with `from_pyramid`, in place of decoding them with
      `read_rgb` outside it. The cache decodes with the same `read_rgb` into a
      pyramid of the same `full_levels` depth, so the refinement reads the
      same pixels and the file is identical bit for bit; a photograph already
      in the cache is not decoded again, and what the build decodes stays
      there for the panels and the bench's steps. `get_many` asks for each
      path once and the cache evicts no decode in flight, so the build
      decodes no photograph twice; the pyramids the cache keeps are the ones
      the build holds, so its peak is no higher than before, and afterwards
      the cache keeps them within its budget. A photograph the cache refuses
      fails the build naming it with the cache's reason, which `get` answers
      from the remembered failure, since `get_many` drops it: a missing or
      undecodable one as before, and now also one with a side over 65535 px,
      refused from its header. The `read photographs` phase notes how many
      were read from disk and reused from the cache, and the cache's
      occupancy, as the bench's `decode images` does, and counts `images`
      where it counted `image`. Specified in
      [gui/index-files.md](../gui/index-files.md) and
      [core/camera/photograph-cache.md](../core/camera/photograph-cache.md),
      with tests in
      [index_files/tests.rs](../../crates/sfm-explorer/src/index_files/tests.rs).
    - **11c, the point order.** **Built**, so step 11 is complete. The
      bulk kernels that loop over points (`render_patch_cloud_bitmaps` under
      the display bitmaps, the conversion and the web export, and under
      `PatchCloud.render_bitmaps`; `refine_normals`; `refit_points`) take
      them in the order of the companion's § "Memory", `BulkPointOrder` in
      [reconstruction/bulk_point_order.rs](../../crates/sfmtool-core/src/reconstruction/bulk_point_order.rs),
      their parallel loop running over the ordered list, and write each
      result at its point's index, so results stay identical bit for bit;
      the whole-cloud render's error is still the first patch's that failed,
      in the cloud's order. The order is the one stored reference
      observation `PointSet::reference_observations` names, including the
      viewer's display-only picks, since the kernels render from that one; a
      reference that is not within its track counts as none. The
      cluster-patches build is not one of these: it loops over clusters of
      `.sift` features, which have no reconstruction point or stored
      reference, so it keeps its cluster order. Specified in
      [core/camera/view-region-pyramids.md](../core/camera/view-region-pyramids.md),
      [core/patch/reference-view.md](../core/patch/reference-view.md),
      [core/bench/refine-normals.md](../core/bench/refine-normals.md) and
      [core/bench/refit-points.md](../core/bench/refit-points.md), with tests
      in
      [bulk_point_order/tests.rs](../../crates/sfmtool-core/src/reconstruction/bulk_point_order/tests.rs)
      (the order of a synthetic point set, with ties, points with no
      reference and a point set with no keypoints),
      [patch/region_read_tests.rs](../../crates/sfmtool-core/src/patch/region_read_tests.rs),
      [bench/tests/refine_normals.rs](../../crates/sfmtool-core/src/bench/tests/refine_normals.rs)
      and [bench/tests/refit_points.rs](../../crates/sfmtool-core/src/bench/tests/refit_points.rs)
      (each kernel's answer in the order equal to its answer over the
      points by index on one thread, and, on four threads with a pool of
      half what the run by index held at once, the pool never exceeded,
      every read counted in it, and some reads copied and others taken
      through the tiles). That the order decodes fewer tiles is tested in step
      13e5.

    Each of 11a to 11c reports in its pull request, for the kernels it
    changes, the count of renders that took the tile path on the test data
    and on a real capture, as the companion's step 5 asks.
12. **The nearby-track grey reads by region** (companion step 6), with the
    step-12 signature changes and their bindings. **Built.** Each grey read
    of the patch read along a ray, under the far-field sweep and the depth
    layers, works out from the photograph's size the four pixels each of its
    samples takes, reads nothing when one is off the photograph, and
    otherwise reads the bounding box of those pixels, grown by the blur's
    4 px and clipped to the photograph, with `read_source_rect` from the
    caller's pool, and makes grey and blurs only the bounding box. The blur
    reflects at the photograph's edges alone, so every sample equals the one
    the whole photograph's blurred grey gives, bit for bit, and the search's
    results are identical. `GreyImages`, the per-view cache of a blurred grey
    copy of each photograph, and the copy each Python `ImagePyramidSet` kept
    are gone; a read holds its rectangle and the grey of it, a few kilobytes,
    and makes each rectangle again when a later read needs it, since the
    grey reads use no `ViewRegions`. The signatures change as the table in
    § "Errors from reading pixels" says. The bindings of
    `read_patch_along_ray` and `depth_layers` make a pool for the call and
    raise a read that fails as `RuntimeError` in place of the interim
    `ValueError`, as those of the sweep and the search already did; the
    bindings in `bench/nearby.rs` keep `ValueError`, since the sources they
    run read no photograph. `held_pyramid` is left only to the tests that
    compare a read with the whole pyramid's. The viewer's *Find Nearby
    Tracks* `read regions` note, written since 10a, now counts the grey
    reads too. Specified in
    [core/bench/far-field-sweep.md](../core/bench/far-field-sweep.md),
    [core/bench/depth-layers.md](../core/bench/depth-layers.md),
    [core/bench/nearby-tracks.md](../core/bench/nearby-tracks.md) and
    [core/camera/view-region-pyramids.md](../core/camera/view-region-pyramids.md),
    with tests in
    [bench/nearby/region_read_tests.rs](../../crates/sfmtool-core/src/bench/nearby/region_read_tests.rs)
    (a read's samples equal the whole photograph's blurred grey's in the
    middle and at each corner, on either path; the patch read, the depth
    layers, the sweep and the search identical with every read copied, every
    read through 32 px tiles and a pool of half the first run's peak; and a
    photograph missing pixels making each return its photograph error, read
    for the template and in the loop over the images).
13. **The tiled photograph cache.** The cache hands out `Arc<TiledPhotograph>`
    for every photograph and keeps no whole pyramid: an in-core photograph is
    decoded whole into tiles, the requested tiles kept as used and the rest
    as unused, by a kernel's `get_many` in its read phase (its misses in
    groups sized by the build budget), by `get`, by a render only for a tile
    evicted during its kernel, and by a panel's one `try_decode_tiles` task
    per photograph; one budget for tiles and file bytes
    (`budget_bytes`, the pyramid budget's default and variable); the layout
    states; the slot as a mutex and condition variable with cancellation
    between a decode and its cutting; the cache's build budget and its count
    (`PhotographCache::reserve_builds` for kernel callers,
    `try_reserve_build` for panels), which in-core whole decodes reserve
    from; `get_level`; `display_layout`, `display_tiles` and `display_level`
    in place of `display_photograph`, so Image Detail, the 3D background and
    every other reader read tiles from the cache; Image Detail asking for the next image's tiles a frame ahead while
    animation plays, and `get_image_detail_view`'s `level` and
    `tiles_pending`; `from_pyramid` deleted.

    The step is thirteen pull requests, 13a to 13e, with 13c in three,
    13d in two and 13e in six. Once the
    cache keeps tiles in place of pyramids, no reader can be handed a whole
    pyramid, so every reader that still takes one (Image Detail, the 3D
    background, Track View, the Track Table, the bulk callers that wrap with
    `from_pyramid`) has to move before that, and the slot and the build
    budget, which every decode goes through, come first. So the split runs
    from the slot up: 13a makes the slot cancellable, 13b bounds the decodes
    in progress by the build budget, 13c1 adds the cache's layout and tile
    requests, 13c2 moves Image Detail to them, 13c3 the 3D background,
    13d1 Track View, and 13d2 the Track Table and Measurement View's
    endpoint crops, while the cache
    still holds whole pyramids and cuts each tile
    it hands out from one, and 13e replaces the pyramids with tiles, which
    every reader then reads, in the five parts its entry gives. Until 13e
    the cache decodes as it does today, so
    every result and every picture is identical after each of 13a to 13d;
    13e keeps every result identical too, since an in-core photograph's tiles
    hold its pyramid's pixels.

    - **13a, the slot.** **Built.** A path's slot is a mutex over its state
      (empty, being made by one caller, or made, with the pyramid or the
      remembered failure) and a condition variable, in place of a
      `OnceLock`. `PhotographCache::get` takes the caller's `&Progress`,
      which it polls for cancellation and reports nothing through, and
      `GetError` gains `Cancelled`, which is never remembered; `get_many`
      passes its own `Progress` down. A caller that finds the slot being
      made waits with `Condvar::wait_timeout` of `SLOT_WAIT_POLL` (50 ms),
      polling its `Progress`, and stops with `Cancelled`, leaving the decode
      to its maker. The maker polls once between the decode and building the
      pyramid, which 13e replaces with cutting the tiles; a cancelled maker
      builds nothing, sets the slot back to empty and wakes the waiters, and
      the first waiter to take the lock decodes in its place. A guard
      empties the slot and wakes the waiters when the maker unwinds. The
      callers that have a `Progress` pass it (the cluster-patches build's
      `get` for the reason a photograph failed, which reports a cancelled
      wait as the build's cancellation, and the reference picks' `get` per
      observation, after which a cancelled pick sends nothing rather than a
      point without a pick); the panels' decode task passes
      `&Progress::none()`.
      Specified in
      [core/camera/photograph-cache.md](../core/camera/photograph-cache.md),
      with tests in
      [photograph_cache/tests.rs](../../crates/sfmtool-core/src/camera/photograph_cache/tests.rs),
      each waiting for the threads it starts with a time limit, so a caller
      that never returns fails the test rather than hang it: a caller
      cancelled mid-decode stops before the pyramid and leaves the slot
      empty, and a waiter on that slot then decodes; a `get` and a
      `get_many` waiting on a decode held in progress stop when they are
      cancelled, and the decode goes on for its maker; a maker that panics
      leaves the slot empty for the next caller; and a cancelled decode is
      not remembered, with a budget or without one. The tests do not time
      the 50 ms, since a time measured across threads depends on how they
      are scheduled; they hold the decode a waiter waits on until the waiter
      has stopped, so only its own polling can stop it. A test in
      [track_table_panel/tests.rs](../../crates/sfm-explorer/src/track_table_panel/tests.rs)
      cancels a reference pick as it reports fetching its last photograph,
      and the pick gives no answer rather than a point without a pick.
    - **13b, the build budget.** **Built.** `PhotographCacheOptions`
      (`budget_bytes`, `build_budget_bytes`, and `tile_size`, which only 13e
      reads), `PhotographCache::new(options)`, `default_build_budget_bytes`
      (decided by the user: the cache budget's rule, a quarter of physical
      memory clamped to 1 to 16 GiB, 4 GiB where it cannot be read,
      overridden in megabytes by `SFM_EXPLORER_BUILD_BUDGET_MB`, whose 0
      lets each caller decode one photograph at a time; the companion's open
      question 8), `reserve_builds`, `try_reserve_build` and
      `BuildReservation`. A whole decode reserves 5 bytes per pixel of the
      photograph. `get` reserves its one decode with `reserve_builds` when it
      marks the slot as being made, from an empty slot or one a cancelled
      maker has just emptied, once it has read the header for the size, and
      gives it back when it ends, is cancelled or unwinds; a maker gives its
      share back before its slot changes state; `get_many` reads each path's
      header first, to know its misses and their sizes, then decodes the
      misses in groups sized by the build budget, one `par_iter` a group,
      each member reserved at its group's largest size, a member that finds
      its slot being made by another caller keeping its share while it
      waits; `stats` gains the build budget, the bytes reserved and the whole
      decodes. A cache of budget 0 reserves its decodes too. Every caller of
      `PhotographCache::new` takes `PhotographCacheOptions`: the viewer's
      session cache its `Default`, and the web export's budget-0 cache and
      the tests' caches their budget with the default build budget. Still
      whole pyramids, and every result identical.
      Specified in
      [core/camera/photograph-cache.md](../core/camera/photograph-cache.md)
      § "The build budget", with tests in
      [photograph_cache/tests.rs](../../crates/sfmtool-core/src/camera/photograph_cache/tests.rs),
      which read the count through the test build's hook and assert only
      what holds under every order the threads run in: a `get_many` over 20
      photographs decodes them in groups that never exceed the build budget
      (set to three decodes), every decode reserved as it starts, with a
      budget and with a budget of 0; four `get_many` callers on one cache
      started together on separate threads never reserve more than the build
      budget plus one decode each; `reserve_builds` grants what fits and one
      when none fits, and `try_reserve_build` takes the count past the
      build budget only alone, from one thread or eight (restated by step
      13c2's revision, which grants a decode that does not fit when nothing
      is reserved); two caches in one process keep
      separate counts, and their decodes never add to the count beyond what
      was reserved; a caller cancelled mid-decode gives its share back, and
      the waiter that decodes in its place reserves, a `get` holding nothing
      while it waits and a `get_many` member keeping its share; and a maker
      that panics gives its share back.
    - **13c1, the cache's layout and tile requests.** **Built.** `layout`,
      `request_layout` and `PeekedLayout` (whose `Building` comes with step
      14), `peek_tile` and `PeekedTile` (whose `Building` comes with step 14
      too, since only a photograph that is `Building` answers it),
      `get_tile`, `view_region_pyramid`, `get_level`, `try_decode_tiles`
      and `TryDecode`, and `TileError` (whose `Changed` comes with step 14,
      since only mip level files can change under a reader), over the whole
      pyramids the cache still holds: a tile, a region or a level is cut from
      the held pyramid when it is asked for, every layout is in core, and
      `PhotographCacheOptions::tile_size` sets the tiles' side. Two changes
      to the interface the plan gave, found while building it:
      `TryDecode::Decoded` carries the tiles asked for, since under a budget
      of 0 the cache keeps nothing and a panel would otherwise never see
      what its task decoded; and `TileError` gains `OutOfRange`, for a level,
      tile or rectangle the photograph does not have, so a caller's bad key
      is an error rather than a panic. Layouts are kept whatever the budget,
      outside it, so a panel under a budget of 0 learns a layout once.
      `try_decode_tiles` reads the header, refuses a key the photograph does
      not have before it reserves, reserves with `try_reserve_build`, and
      claims the slot only when no other caller holds it, answering `Busy`
      otherwise; a header that cannot be read is remembered in the slot as
      `get` remembers it. `try_decode_tiles`, `get_tile`,
      `view_region_pyramid` and `get_level` take the caller's `&Progress`,
      since an in-core miss decodes the whole photograph, and a cancelled
      call answers `TileError::Cancelled`, never remembered, with its
      reservation given back. No panel calls them yet, so nothing drawn
      changes.
      Specified in
      [core/camera/photograph-cache.md](../core/camera/photograph-cache.md)
      § "Layouts and tiles", with tests in
      [photograph_cache/tiles/tests.rs](../../crates/sfmtool-core/src/camera/photograph_cache/tiles/tests.rs):
      every tile, a region and every level equal the held pyramid's pixels,
      each tile with its border as Image Detail cuts it today; what the
      photograph does not have is `OutOfRange`; a panel decode whose
      `try_reserve_build` is refused (the budget used up by another caller's
      reservation) decodes and reserves nothing, and decodes once the
      reservation is given back; a panel decode that finds a slot another
      caller is decoding answers `Busy` at once, never waiting on it (a decode
      held in progress through the test hook, with the panel's thread bounded
      by a time limit); one that loses the slot after reserving gives its
      reservation back; and panel decodes from eight threads never take the
      count past the build budget, with a budget and with a budget of 0;
      and each of the four calls, cancelled during its decode, answers
      `Cancelled` with nothing reserved or remembered, and the next call
      decodes.
    - **13c2, Image Detail's requests by layout and tile.** **Built.**
      `display_layout` and `display_tiles` replace `display_photograph` for
      Image Detail, which draws the tiles the cache hands it in place of
      cutting them itself, asks for the next image's tiles a frame ahead
      while animation plays (`prefetch_tiles`, through
      `ImageDetail::prefetch`, at the view the panel carries over to that
      image), and reports `level` and `tiles_pending` in
      `get_image_detail_view`. `display_layout` acts on `layout`, starting
      one `request_layout` task for a photograph that is `NotKnown`, and
      `display_tiles` on `peek_tile`, starting one `try_decode_tiles` task per
      photograph with the tiles it is missing, at most one in flight per
      photograph, the tiles that task answers kept in `PhotographRequests`
      until the panel asks for them, which is how a budget of 0 sees them.
      The panel asks only for the tiles it uploads on the frame (at most 16)
      and never for one it has uploaded, so `peek_tile`'s copy of a tile
      happens once per upload, off the frames that change nothing. The panel
      draws the same pixels as before: the textures it uploads and the meshes
      it paints, over test-data photographs at the fit and zoomed in, were
      compared with those of the panel before this step and are identical.
      Six things the plan did not say, found while building it: the
      functions are named `display_layout` and `display_tiles`, since the
      layout and the tiles are two questions and a frame asks for its tiles
      together, which is what lets one task decode them; `layout` keeps a
      held photograph's layout beside the ones `request_layout` reads and
      answers the same `Arc` while its size stays the same, so a panel asking
      every frame allocates nothing; a decode that answers `Busy` repaints
      50 ms later, so the panel asks again then rather than on every frame;
      a decode's failure is kept beside the answered tiles, with the file's
      length and modification time, so under a budget of 0, which remembers
      no failure, the panel says *Failed to load image* rather than decode
      again until a check of the file on the rayon pool finds it changed; each
      ask for a photograph drops its kept tiles the ask does not name, so a
      photograph keeps at most one decode's answer (16 tiles), and the
      answers are kept for the four photographs answered last, about 51 MB at
      most; and a key the photograph does not have,
      which only a layout read before the file changed size names, makes the
      task `forget` the path, so the next frame reads the header again.
      A newly selected image is drawn once its overview level's tiles have
      landed; until then Image Detail keeps drawing the image it showed
      before, its tiles, overlays and view, taking no click, drag or menu (G13),
      and says *Loading image…* only where it has no such image (decided by
      the user). `display_photograph` stays for the 3D background
      (13c3), Track View's and the Track Table's renders and the dock's
      pre-caching of the viewed and bench tracks' photographs (13d).
      Specified in
      [gui/multi-panel-image-browser.md](../gui/multi-panel-image-browser.md)
      § "Image Detail",
      [gui/mcp-server.md](../gui/mcp-server.md) § "The Image Detail view" and
      [core/camera/photograph-cache.md](../core/camera/photograph-cache.md)
      § "In the viewer", with tests in
      [state/photograph_requests/tests.rs](../../crates/sfm-explorer/src/state/photograph_requests/tests.rs)
      and
      [image_detail/photograph_tiles/tests.rs](../../crates/sfm-explorer/src/image_detail/photograph_tiles/tests.rs),
      whose tasks wait in a queue the test runs on a thread of its own,
      bounded by a time limit, so no assertion depends on how threads are
      scheduled: two panels needing eleven tiles of one photograph (ten of
      level 0 and the overview) start one header read and one decode, the
      second panel starting none while the first's waits, and the frames
      before it lands draw no image; a decode whose
      `try_reserve_build` is refused (the build budget held by another
      reservation) decodes nothing, ends within the time limit while the
      other reservation is still held, so no rayon worker waits on the count,
      is asked again on the next look and decodes once the reservation is
      given back; no panel decodes on the GUI thread, the decode count
      staying 0 until the queued tasks run, with a budget and with a budget
      of 0; a failed decode is answered as unreadable and not decoded again
      until its file changes, a check of the file finding the change; tiles
      answered for a view that moved on are dropped, so a panel panning
      while its decodes run keeps at most one decode's answer; a key the
      photograph does not have reads the layout again; the tiles asked for
      ahead of an image switch, zoomed in and panned, are the ones the next
      image's first frame draws, with no task started on the switch; a
      switch keeps drawing the previous image, its tiles and view and no
      *Loading image…*, until the new image's overview has landed, then
      draws the new image with the look asked for on the switch; and the
      first image says *Loading image…* until its tiles land. The core's
      [photograph_cache/tiles/tests.rs](../../crates/sfmtool-core/src/camera/photograph_cache/tiles/tests.rs)
      test the kept layout, and
      [mcp/tests/render.rs](../../crates/sfm-explorer/src/mcp/tests/render.rs)
      the two fields. A panel decode that finds the slot held by another
      caller answers `Busy` the same way; that is tested in the core, since
      only the core's test build can hold a decode in progress.
    - **13c3, the 3D background's level.** **Built.** In place of
      `display_photograph`, the 3D background asks for its photograph's
      layout with `display_layout`, chooses its level from the layout's level
      sizes, and asks for the level with `display_level`, which starts one
      task on the rayon pool. The task first asks for the level's tiles with
      `try_decode_tiles`, ending at once when the answer is `Busy`, after
      which the background asks again 50 ms later, and builds the level from
      the tiles that call returns, at every cache budget, never calling
      `get_level`. So no panel task waits on a slot or the count, and the
      rule has no exception (decided by the user, the companion's open
      question 9). The plan first read the level with `get_level` once the
      tiles were present, under a budget other than 0; that call could find
      the photograph evicted by another decode since the tiles were cut and
      wait on its slot while another caller decoded it, so the user decided,
      once building it found that gap, that the level is built from the
      tiles at every budget. The background draws the same pixels as
      before: the level, its size, the texture coordinates' scale and the
      RGBA bytes it uploads, for one photograph of each of the four test-data
      sets in viewports from 16 to 3840 px wide, with a budget and with a
      budget of 0, were compared
      with those of the background before this step and are identical. Five
      things the plan did not say, found while building it: `upload_bg_image`
      takes two closures, the layout's and the level's, since the level is
      chosen from the layout before it is asked for; the level task shares
      Image Detail's limit of one task per photograph and its kept failures
      in `PhotographRequests`, so a photograph whose decode failed is not
      decoded again until its file changes, under a budget of 0 too, where
      the background read it again on every frame before; the level it
      answers is kept apart from the tiles, one level at most, handed over
      once and dropped when another photograph or level is asked for, since
      a level is many tiles (48 for level 2 of a 200-megapixel photograph,
      37 MB as RGB, and at most 151 MB under the texture limit of 8192) and
      would break the bound of 16 kept tiles a photograph; a level read
      against a layout the file no longer has, whose tiles are not the ones
      the layout places, makes the task `forget` the path, as a tile key the
      photograph does not have does; and
      while the level loads the previous photograph stays on its own mesh,
      as before, since until the tiled cache the coarsest level comes from
      the same decode as the level drawn. What the user can see change is
      timing alone: under a budget of 0 the decode no longer stalls the GUI
      thread, and the photograph is drawn on a later frame than the one that
      asked for it; with a budget, a photograph the cache holds is drawn on
      the frame after its level task runs rather than on the frame that asks.
      `display_photograph` stays for Track View's and the Track Table's
      renders and the dock's pre-caching of the viewed and bench tracks'
      photographs (13d). Specified in
      [gui/camera-views.md](../gui/camera-views.md) § "Photograph
      background" and
      [core/camera/photograph-cache.md](../core/camera/photograph-cache.md)
      § "In the viewer", with tests in
      [state/photograph_requests/tests.rs](../../crates/sfm-explorer/src/state/photograph_requests/tests.rs)
      and
      [scene_renderer/upload/tests.rs](../../crates/sfm-explorer/src/scene_renderer/upload/tests.rs),
      whose tasks wait in a queue the test runs on a thread of its own,
      bounded by a time limit: every level of a photograph whose levels cross
      several tiles is read by one task however often it is asked for, none
      decoded on the GUI thread, and built from the tiles equals the level
      of the pyramid the background drew before, with a budget and with a
      budget of 0, the task making one call to the cache, so a level read of
      a photograph the cache holds counts one hit, where a `get_level` after
      the tiles would count two; a level
      read whose `try_reserve_build` is refused answers `Busy`, decodes
      nothing, is asked again on a later frame and is read once the
      reservation is given back, with a budget and with a budget of 0; a
      level read waits behind Image Detail's decode of the same photograph,
      and a level not taken is dropped when another is asked for; a level
      read against a layout the file no longer has, wider so a tile does not
      fit or narrower so the tiles would leave a column unset, or of a level
      the layout does not have, reads the layout again; a failed level read
      is answered as unreadable and not decoded again; and the background,
      drawn through `display_layout` and `display_level`, asks for the header
      and the level
      once each, decodes nothing on the GUI thread and draws the photograph
      once it lands, keeps the previous photograph while the layout or the
      level loads, and clears it for an unreadable layout or level. Each of
      these tests takes 0.03 s or less.
    - **13d1, Track View.** **Built.** Track View's patch tiles,
      cluster-stage tiles and hover views, and its crops and their hover
      views, act on the layout state and peek tiles, asking for the missing
      ones with `try_decode_tiles`, and render only once every tile they
      need is there. A render reads from a photograph that holds exactly the
      tiles the picture peeked, `TiledPhotograph::from_tiles`, pinned for the
      render, so a tile evicted between the peek and the render cannot be
      read and the GUI thread never decodes (the companion's open question
      7). The dock no longer asks for the bench track's and the viewed
      track's photographs: each picture asks for its own tiles as it is
      drawn, `read_peeked` in
      [state/photograph_requests.rs](../../crates/sfm-explorer/src/state/photograph_requests.rs),
      so the dock fills only the SIFT cache for the viewed track. The
      pictures are the same pixels as before: every tile, crop and hover view
      of two points of each of `seoul_bull_sculpture_ground_truth_v02`,
      `kerry_park_ground_truth_v02` and `dino_dog_toy_ground_truth_v01`, at
      the track and the cluster stage, drawn by whole headless frames, were
      compared with those of the panel before this step and are identical,
      with the photographs in the cache and, after this step, under a cache
      that keeps nothing, where before this step Track View drew none of
      them. Five things the plan did not say, found while building it: a
      read of a photograph made of tiles is refused with
      `PhotographReadError::NeedsTiles`, naming every tile the read needs,
      before any pixel is read, which is how a picture learns the tiles of its
      regions, from a render on the coarsest level's first tile alone, with
      no second copy of the region logic in the panel; a photograph made of
      tiles needs at least one, since its channels are its tiles', which is
      why that first tile is always among a picture's tiles; a picture's
      tiles are handed over together or not at all (`display_tile_set`), and
      none of its asks drops another's answer, since several pictures read
      one photograph and a picture that took part of its tiles would, under
      a budget of 0, lose them before the rest landed; the answers so kept
      for a photograph are bounded at 32 tiles, an answer that would take
      them past it replacing them, so the four photographs answered last keep
      about 101 MB at most while no one render needs more than 32; and a photograph that cannot be read, which
      before this step left the row's cells empty for ever, draws the cells'
      error state, *unreadable* with why on hover. Specified in
      [gui/track-view.md](../gui/track-view.md),
      [core/camera/view-region-pyramids.md](../core/camera/view-region-pyramids.md)
      and [core/camera/photograph-cache.md](../core/camera/photograph-cache.md)
      § "In the viewer", with tests in
      [sampler/source/held_tiles_tests.rs](../../crates/sfmtool-core/src/camera/sampler/source/held_tiles_tests.rs):
      every render, grid, sample, rectangle and clipped-share read of a
      photograph made of
      the coarsest first tile alone is refused naming the tiles it needs, the
      photograph made of those tiles less any one is refused naming the same
      tiles, and the photograph made of them reads the whole pyramid's values
      bit for bit, through the pool and through tiles; and `from_tiles`
      refuses no tiles and a tile moved, resized, past the layout's levels or
      columns, of the wrong pixel size or of other channels;
      [state/photograph_requests/tests.rs](../../crates/sfm-explorer/src/state/photograph_requests/tests.rs),
      whose tasks wait in a queue the test runs by hand: a render waits,
      rendering nothing, while the header and then the first tile are read,
      each asked for once over three looks, learns its twelve tiles and runs
      on them with the photograph's own pixels, nothing decoded on the
      calling thread, with a budget and with a budget of 0; a photograph that
      cannot be read is answered as unreadable, its render never run; two
      renders of one photograph and Image Detail asking for others of its
      tiles all get them under a budget of 0, nothing left kept; a render's
      set of tiles partly answered takes none of them, leaving the answered
      ones kept while the rest are decoded, and is handed over whole after
      that decode; and a photograph keeps at most 32 answered tiles, past which the newest
      answer replaces them; and
      [track_view/body/tests/region_reads.rs](../../crates/sfm-explorer/src/track_view/body/tests/region_reads.rs):
      the panel's tests of its pictures, which compare them with the whole
      pyramid's, pass unchanged; a row whose photograph the cache does not
      hold draws its cells' loading state, asking for its layout once over
      three frames and decoding nothing, and draws its pictures on the frame
      after the photograph lands; a row whose photograph file cannot be
      read draws *unreadable*; and a hover view whose tiles are not all
      there says *Loading the photograph…*, asks once and is drawn once
      they are. The core's tests take 0.19 s, the first, and 0.01 s or less;
      the request tests 0.02 s or less; the panel's three about 0.17 s each.
    - **13d2, the Track Table and Measurement View.** **Built.** The Track
      Table's crop column and hover, and Measurement View's endpoint crops
      and their hover views, which the plan did not list but which also took
      whole pyramids from `display_photograph`, read through `read_peeked`
      as Track View does, so no panel reads a whole pyramid any more and
      none decodes on the GUI thread, at any budget; `display_photograph` is
      deleted, and the viewer's `peek_full_res_pyramid` is gone. `peek`
      stays for the one viewer caller that only asks whether a photograph is
      decoded, the image menu, which offers *Add image to tracks* for an
      image whose photograph the cache holds or whose file exists, and
      answers tiled photographs from 13e. The plan said `peek_state` stays
      for that caller too; the image menu calls `peek`, and `peek_state`,
      whose viewer callers were `display_photograph` and the Track Table's
      look while its rows scroll, has no caller in the viewer now and stays
      in the core. The pictures are the same pixels as before: the crops of
      the first 29 rows of each of `seoul_bull_sculpture_ground_truth_v02`,
      `kerry_park_ground_truth_v02` and `dino_dog_toy_ground_truth_v01` and
      the hover views of the first 8, drawn by whole headless frames, and
      both endpoints' crops and hover views of the six artificial
      measurements that
      [scripts/measurements/add_artificial_measurements.py](../../scripts/measurements/add_artificial_measurements.py)
      writes to a copy of each, were compared with those of the panels
      before this step and are identical, with the photographs in the cache
      and under a cache that keeps nothing, where before this step the Track
      Table's hover views were never drawn under such a cache (they said
      *The photograph is being read again.*). Four things the plan did not
      say, found while building it: the Track Table's rows ask for nothing
      while they scroll (§ "Requests wait for the scroll to stop" in
      [gui/track-table.md](../gui/track-table.md)), so while they scroll its
      crops and their hover views render through
      `read_peeked_without_asking`, which runs the
      render on what the cache and the kept answers hold and starts no
      header read, no decode and no check of a failed file; Measurement View
      cuts an endpoint's crop and its hover view in one render whose refusal
      names the tiles of both, so they are asked for together; the hover
      of a Track Table crop whose tiles are loading says *Loading the
      photograph…*, as Track View's do, and one whose hover view cuts
      nothing says *Nothing around this crop could be rendered.*, as Track
      View's crop hover does, in place of *The photograph is being read
      again.*, which said both; and the renders stay inside the 32 kept
      tiles: a crop reads at most 128 texels a side at a level whose pixel
      is not wider than its step, under 258 pixels of 512-pixel tiles, so at
      most four tiles and the first, and a hover view three times as wide at
      most nine, which hold the crop's, so a Measurement View render needs
      at most ten. A photograph that cannot be read is the one answer no
      panel keeps: Track View, the Track Table and Measurement View ask
      again for it on every frame, so a picture is drawn once a header read
      that ended mid-frame has landed, and the failed decodes are kept apart
      from the answered tiles, for every photograph, so asking again decodes
      nothing until the file changes.
      Specified in [gui/track-table.md](../gui/track-table.md),
      [gui/measurement-view.md](../gui/measurement-view.md) and
      [core/camera/photograph-cache.md](../core/camera/photograph-cache.md)
      § "In the viewer", with tests in
      [state/photograph_requests/tests.rs](../../crates/sfm-explorer/src/state/photograph_requests/tests.rs):
      a render that asks for nothing waits on a photograph nothing asked
      for, starting nothing over three looks, runs on the photograph's own
      pixels once `read_peeked` has had it read, and answers a photograph
      whose decode failed under a cache that keeps nothing as unreadable
      without starting a check of its file;
      [track_table_panel/tests/crop.rs](../../crates/sfm-explorer/src/track_table_panel/tests/crop.rs),
      whose tests of the crops and their hover views against the whole
      pyramid's pass unchanged: a crop whose tiles are not there draws an
      empty frame, asks for its photograph's header once over three frames,
      decodes nothing and draws once the photograph lands, and a hover view
      whose photograph has left the cache says *Loading the photograph…*,
      asks once and is drawn once it is back, and one whose photograph can
      no longer be read says why; and
      [measurement_view/tests/crops.rs](../../crates/sfm-explorer/src/measurement_view/tests/crops.rs):
      both endpoints' crops and hover views, read from photographs on disk
      with a budget and with a budget of 0, equal the whole pyramid's, say
      *loading…* until they land with each photograph asked for once over
      three frames and nothing decoded on the GUI thread, take two rounds of
      reads with a budget and three with a budget of 0, a crop and a hover
      view that need different tiles being asked for together, and a
      photograph that cannot be read gives *no crop*. The request test takes
      0.01 s, the Track Table's three 0.08 s to 0.11 s, and Measurement
      View's 0.10 s and 0.01 s.
    - **13e, tiles in the cache.** The cache keeps photograph tiles in place
      of pyramids, the tiles a request asked for as used and the rest of the
      decode as unused, evicted first, under the one budget; `get`,
      `get_many`, `peek` and `insert` answer or take `Arc<TiledPhotograph>`
      and `ImageU8`; a tiled photograph from the cache reads its tiles
      through `get_tile`, decoding an in-core photograph again, with
      `reserve_builds`, for a tile evicted during its kernel; `from_image`
      cuts tiles the source owns, and `from_pyramid` is deleted, with
      `held_pyramid`, so the bulk callers (`ViewSources`, the display
      bitmaps, the cluster-patches build, `sfm web-export`) and the
      bindings take what `get_many` and `from_image` answer;
      `PhotographReadError` gains `Changed`, `NotFound`, `TooLarge` and
      `Unreadable` with the conversions from `TileError` and `GetError`, and
      `sfm web-export` reports the `RuntimeError` a read that fails raises
      through `write_web_export` as its failure, as it reports the
      `ValueError` of a refusal, since a read can fail from here on.

      It is six pull requests, 13e1 to 13e5 with 13e4 in two, each leaving
      every result identical. The answers' type comes first, while the cache
      still holds whole pyramids, so every caller takes what the cache answers
      before what the cache holds changes; then the tiles a source owns, so
      that nothing but the cache reads a pyramid in place; then the cache's
      own tiles, with each answer holding every tile of its photograph as a
      caller holds a pyramid today; then the answers that hold no tile and
      read through the cache, which is the part a read can fail in, so the
      errors, and `sfm web-export`'s report of a failed read, come with it;
      and last the tests of the bulk kernels under a small budget, with a
      render's decode kept whole while its photograph is under half the
      budget.

      - **13e1, the answers as tiled photographs.** **Built.** `get`,
        `get_many`, `peek` and `peek_state` answer `Arc<TiledPhotograph>`.
        The cache still decodes and holds whole pyramids: it makes one
        tiled photograph of each pyramid as it builds it, or as `insert` is
        handed one, cut into tiles of the cache's `tile_size` and reading
        the pyramid in place, and its slot holds that photograph, which
        every answer is and every tile request reads. `insert` still takes
        an `Arc<ImageU8Pyramid>`. The callers that wrapped what the cache
        answered with `from_pyramid` take the answer as it is:
        `ViewSources`, whose `DecodedViews` holds the answers and whose
        placeholder is `from_image` of one pixel; the display bitmaps, and
        so `sfm web-export`; the cluster-patches build, which hands the
        answers to `refine_cluster_patches_borrowed`; the Track Table's
        reference picks, whose size check reads the layout; and the
        `align_vs_congealing` and `pairwise_flow_measure` examples. Every
        kernel reads the same pyramid through the same 512 px tiles as
        before, so no result changes: on
        `seoul_bull_sculpture_ground_truth_v02`,
        `kerry_park_ground_truth_v02` and `dino_dog_toy_ground_truth_v01`,
        the display bitmaps and their reference observations, with a budget
        and with a budget of 0, and the reference-view rule's picks, tiles
        and readings for the first 40 points, as the Track Table's picks read
        them, are the same bit for bit as before this step. Specified in
        [core/camera/photograph-cache.md](../core/camera/photograph-cache.md),
        with a test in
        [photograph_cache/tiles/tests.rs](../../crates/sfmtool-core/src/camera/photograph_cache/tiles/tests.rs):
        the photograph `peek` answers has the cache's layout and tile size
        (16 px in the test), its pyramid is the one the tiles are cut from,
        and a `get` answers the same `Arc`, as a hit. The tests that compare
        a read with the whole pyramid read it through `held_pyramid` in the
        core and, in the viewer, through `AppState::cached_pyramid`, which
        reads each level whole with `get_level`.
      - **13e2, tiles the source owns.** **Built.** `from_image` holds
        every tile of its layout of the pyramid `ImageU8Pyramid::from_image`
        builds, each with its border, as `from_tiles` holds them; a tile is
        cut by the same clamped copy `from_pyramid`'s tile read makes, so
        its pixels are the same. It never holds that pyramid whole: it cuts
        one level at a time, makes the next from it with `downsample_2x` and
        the pyramid's stop rule, and drops it, so an image being cut peaks at
        about 2.25 times its bytes (its level-0 tiles, level 0 and level 1);
        step 13e3 makes each next level from the tiles just cut, which
        brings the peak to about 2.0 times.
        The bindings built a copy of each image and its pyramid, about 2.33
        times, so their peak memory is no higher for any thread count;
        building the pyramid and then cutting it would have held the pyramid
        and its tiles, about 2.67 times, more than before once the threads
        number three quarters of the images. The test-only
        `TiledPhotograph::cut` cuts a pyramid of whole levels already built,
        for the tests that build a pyramid of fewer levels. A photograph
        that holds every tile of its layout
        (`from_image`'s, or `from_tiles` handed all of them) skips the check
        for tiles it lacks before a read. `with_tile_size` puts each level
        together from the tiles and cuts it again at another size. The
        bindings (`ImagePyramidSet`'s photographs in
        [patches/views.rs](../../crates/sfmtool-py/src/patches/views.rs),
        `build_photographs_from_image_list` in place of
        `build_pyramids_from_image_list`, and the cluster refinement's in
        [matching/cluster.rs](../../crates/sfmtool-py/src/matching/cluster.rs))
        take `from_image`, and so does the demo scene's
        `tiled_photographs`. The pairwise-flow scene and the tests that built
        a whole pyramid only to read it through `from_pyramid` take
        `from_image` in the viewer, where the pyramid is the image's full
        depth, and in the core `cut` of the pyramid they build, which often
        has fewer levels than the full depth (four in the pairwise-flow and
        bench scenes), so no test's input changes. A test that compares a
        read with that pyramid builds the pyramid itself: the bench scene
        keeps the pyramids it cuts (`Scene::pyramid`), and a test that only
        needs a level's pixels reads it with the crate-private
        `whole_level`. Only the cache and the tests of reading a pyramid in
        place (a pyramid made by `from_regions` that lacks pixels, and the
        whole-pyramid reads `held_tiles_tests` compares with) still call
        `from_pyramid`. No result changes: `sfm embed-patches` and
        `sfm cluster-patches` on the seoul_bull workspace the
        `seoul_bull_workspace` fixture builds, and `refine_cluster_patches`
        on the binding tests' synthetic clusters with and without
        `piecewise`, give every output array the same bit for bit before and
        after this step. Specified in
        [core/camera/view-region-pyramids.md](../core/camera/view-region-pyramids.md),
        with a test in
        [tiled_photograph/tests.rs](../../crates/sfmtool-core/src/camera/tiled_photograph/tests.rs):
        a photograph `from_image` makes holds every tile of the image's
        pyramid, each tile with its border the pyramid's pixels bit for bit,
        and reads each whole level and rectangles across tile edges and at
        the far corner as the pyramid holds them, cut at 512 px over a
        photograph three tiles wide and two high, and a smaller photograph cut
        again at 16 px; and `from_image`'s tiles, cut level by level, equal
        `cut` of the whole pyramid bit for bit, with the same levels, for
        sizes whose pyramid stops short of the full depth and odd sizes over
        many tiles (the two 0.02 s together); the bindings' tests pass
        unchanged.
      - **13e3, the cache keeps tiles.** **Built.** A decoded photograph's slot keeps
        its layout, and the cache keeps its tiles, cut from the decode's
        pyramid, which is then dropped: the tiles a request asked for
        (`try_decode_tiles`' keys, `get_tile`'s key) as used and the rest as
        unused, evicted first, oldest first, under the one budget, an unused
        tile becoming used when a request reads it. `get` and `get_many`
        answer a photograph made of every tile of its decode (`from_tiles`),
        so a caller holds what it holds today, and decode again a
        photograph not every tile of which is cached; `peek` and
        `peek_state` answer one only when every tile is cached. `insert`
        takes an `ImageU8` and cuts its tiles in as unused, and `stats`
        counts tiles. `TiledPhotograph`'s pyramid form, `from_pyramid` and
        `held_pyramid` are deleted; a test that read a pyramid lacking pixels
        reads a photograph made of some of its tiles. Tests: an in-core
        photograph's tiles equal the same pixels of today's pyramid; a
        request for one tile decodes the photograph once and leaves its
        other tiles as unused, evicted first; a tile evicted and asked for
        again decodes the photograph again, counted in `stats`; and under a
        budget of 0 and one smaller than the photograph's level 0, a panel's
        request is answered its tiles and asks nothing again once it has
        them.

        As built, a decode is `ImageU8::read_rgb` and the crate-private
        `cut_image`, which `from_image` uses too: it cuts level 0 at the
        cache's tile size, drops the image and makes each next level from
        the tiles of the one before, bit for bit as `downsample_2x` makes it,
        so the cache never holds a pyramid and its tiles together, and only
        the tiles are kept.
        A slot keeps, behind its mutex, whether a caller is making it, the
        layout or the remembered failure, and its tiles, each with whether
        it is used and its tick; a decode into a slot that keeps some tiles
        leaves those `Arc`s in place and drops its own copies, so a caller
        and the cache share one tile, and the tiles kept stay readable while
        the decode runs. Eviction takes the unused tiles before the used,
        among each the oldest first, and among tiles of one tick the finer
        levels first, and never a tile the request that made the decode
        asked for, which for `get`, `get_many` and `insert` is every tile:
        the rule the cache had for the pyramid just decoded, kept over the
        budget, so a panel's tiles stay in the cache under a budget smaller
        than they are. `peek_tile` and a `get_tile` that finds its tile kept
        mark it used; `get_level` and `view_region_pyramid` read the
        photograph `get` answers. `layout` answers the layout the slot keeps,
        the same `Arc` while it stays the same. `stats` reports `tiles` in
        place of `entries`, and `bytes` counts each tile with its border.
        The tests of a render whose photograph lacks pixels read a
        photograph `from_tiles` makes of some of its tiles, refused with
        `NeedsTiles` where they were refused with `OutOfRange`; the tests in
        which a photograph from the cache lacked pixels (the display bitmaps',
        the open's and the conversion to embedded patches') are gone, since
        no answer of the cache can lack a pixel until step 13e4b, which
        brings them back, and the
        Track Table's and Track View's tests of a photograph that cannot be
        read take one whose header read finds no file.

        Memory: in a release build, `get` of an 8000 x 6000 JPEG (144 MB of
        pixels) raises the process's peak private bytes by 2.13 times the
        image's bytes and keeps 1.34 times, as the decode into a whole
        pyramid did before this step (2.13 times, keeping 1.33): the cutting
        peaks at about 2.0 times, the image and its level-0 tiles, under the
        decoder's own peak. The image menu asks `keeps_every_tile`, which
        allocates nothing and is no use for the eviction order, in place of
        `peek`, since the scene graph lays the menu out for every node on
        every frame. The
        estimate a decode reserves on the build count stays 5 bytes per
        pixel. No result changes: the display bitmaps and their reference
        observations, with a budget and with a budget of 0, and the
        reference-view rule's picks, tiles and readings for the first 40
        points, on `seoul_bull_sculpture_ground_truth_v02`,
        `kerry_park_ground_truth_v02` and `dino_dog_toy_ground_truth_v01`,
        and every output array of `sfm embed-patches` and
        `sfm cluster-patches` on the `seoul_bull_workspace` fixture's
        workspace, are the same bit for bit as before this step. Specified
        in [core/camera/photograph-cache.md](../core/camera/photograph-cache.md),
        with the tests in
        [photograph_cache/tiles/tests.rs](../../crates/sfmtool-core/src/camera/photograph_cache/tiles/tests.rs)
        (the four 13e3 tests take 0.01 s or less each), and a budget smaller
        than the photograph's level 0 added to the viewer's test that Image Detail's layout and tiles are read once
        in
        [state/photograph_requests/tests.rs](../../crates/sfm-explorer/src/state/photograph_requests/tests.rs).
      - **13e4, photographs read through the cache.** `get` and `get_many`
        answer a photograph that holds its layout and the cache's handle on
        its tiles and no pixel: it reads each tile through `get_tile`,
        decoding an in-core photograph again, reserving with
        `reserve_builds`, for a tile evicted during its kernel, so a
        `get_many` holds no pyramid; `get` decodes only a photograph none of
        whose tiles are cached. Every render entry point takes the caller's
        `&Progress`, which such a decode polls for cancellation, and under a
        cache budget of 0 an answer holds the tiles of the decode that made
        it until it is dropped (the companion's open questions 10 and 11).
        `PhotographReadError` gains `Changed`,
        `NotFound`, `TooLarge` and `Unreadable`, with `From<TileError>` and
        `From<GetError>`, which such a read returns, and every caller reports
        them as its failure, as it reports the read errors it reports today:
        `sfm web-export` reports the `RuntimeError` a read that fails raises
        through `write_web_export` as its failure, as it reports the
        `ValueError` of a refusal.

        It is two pull requests. The `&Progress` comes first, alone, since
        it changes the signatures of every render entry point and of the
        kernels between them and their callers, and changes no behaviour
        while every photograph holds its tiles; then the answers that read
        through the cache, with the errors such a read can return, the
        report of a failed read, and the tests of the reads.

        - **13e4a, the render entry points take `&Progress`.** **Built.**
          `render_source_tile`, `render_source_tile_with_grad_into`,
          `render_source_affine`, `sample_source_grid`,
          `read_source_samples`, `read_source_rect`,
          `sample_source_member_grid` and `source_clipped_share` take the
          caller's `progress: &Progress<'_>` as their last argument and hand
          it to every tile read (`TiledPhotograph`'s crate-private `tile` and
          `region`, `copy_regions` and the tile path's window), which a
          photograph that holds its tiles never polls. Every kernel between
          them and a caller that has a `Progress` passes the caller's cancel
          flag down. The public `refine_cluster_patches` and
          `refine_cluster_patches_borrowed`, whose `Option<&AtomicUsize>`
          count of finished clusters is renamed `done`, gain `progress`
          after it; `depth_layers`, `read_patch_along_ray`,
          `refine_view_against_reference`, `render_bitmap_in_place` and
          `spawn_candidate_tracks` gain it as their last argument; and so
          do the crate-private kernels on the way: the cluster refinement's
          member, template and piecewise reads, the sub-pixel refinement's
          renders, the keypoint localizer's last alignment,
          `TrackReferences`' `build`, `search` and `score`, the bench's
          grid-plane, pairwise-flow, evaluation, refit, far-field and
          *Find Nearby Tracks* reads, the steps of `build_track_at_pixel`,
          and `add_image_to_tracks`' measurement and bitmap scores. Where a
          kernel handed a nested call `Progress::none()` so that its reports
          were dropped (each point of `refit_points`' parallel loop, a fit
          inside a far-field refit or inside a step of
          `build_track_at_pixel`, the template renders of `TrackReferences`,
          the localization and refinement inside `spawn_candidate_tracks`),
          it now hands it `progress.silenced()`, new in `sfmtool-progress`,
          which carries the cancel flag and reports nothing, so what a run
          reports is unchanged. A nested call the cancel stops can answer
          as a stage that failed, so each kernel that hands one the flag
          answers its own cancellation rather than a half-built answer:
          `build_track_at_pixel` reads the flag after each member and before
          it answers with a track, `far_field_sweep` before it answers,
          `refit_points` stops on a fit that answers `Cancelled`, and
          `TrackReferences`' `build` and `score` pass a cancelled render on
          as `Cancelled` rather than leaving the view out.
          `constellation_seeds` keeps its signature: it reads no photograph,
          and its descriptor search is quick. The grown plane's step keeps
          the `Progress` in its `StepReads`, beside the failure its renders
          keep, so its objective's trait methods keep their signatures, and
          the view selection passes its `renders` `Progress`, which is the
          caller's in a single selection and in a batch. A panel on the GUI
          thread (Track View's tiles and patches, the Track Table's pictures and
          Measurement View's crops), whose photograph holds the tiles it
          renders, and the bindings that have no cancel flag pass
          `&Progress::none()`, and so do the single-patch helpers
          `localize_patch_keypoints`, `refine_patch_keypoints` and
          `member_zncc_matrix`, which take no `Progress` and have no caller
          that can cancel (each batch caller calls the variant that takes
          one); the viewer's cluster-patches
          build passes its *refine patches* phase. Nothing decodes inside a
          render yet, so no result changes: the display bitmaps and their
          reference observations, the reference-view rule's picks, tiles and
          readings for the first 40 points, and `refine_cluster_patches`
          over every track of the first 40 points, on
          `seoul_bull_sculpture_ground_truth_v02`,
          `kerry_park_ground_truth_v02` and
          `dino_dog_toy_ground_truth_v01`, are the same bit for bit as
          before this step. Specified in
          [core/camera/view-region-pyramids.md](../core/camera/view-region-pyramids.md),
          with a test in
          [sampler/source/tests.rs](../../crates/sfmtool-core/src/camera/sampler/source/tests.rs):
          every entry point given a cancelled `Progress` reads a photograph
          that holds its tiles, through copied regions and through the tile
          path, as it reads it given `Progress::none()`; and in
          [patch/spawn/tests.rs](../../crates/sfmtool-core/src/patch/spawn/tests.rs),
          a cancelled `Progress` stops `spawn_candidate_tracks` with
          `Cancelled`, its flag reaching the localization through
          `silenced`; and in
          [track_at_pixel/tests.rs](../../crates/sfmtool-core/src/bench/track_at_pixel/tests.rs)
          and [nearby/tests.rs](../../crates/sfmtool-core/src/bench/nearby/tests.rs),
          a cancel set from another thread at 40 times spread over a whole
          run makes `build_track_at_pixel` and `far_field_sweep` (on a
          scene whose peak is refitted) answer either `Cancelled` or the
          uncancelled run's answer, never a track or a sweep made of part of
          the work.
        - **13e4b, the answers read through the cache.** **Built.** `get` and
          `get_many` answer a photograph that holds its layout and the cache's
          handle on its tiles and no pixel, each tile read going through
          `get_tile` with the `Progress` the render was given; under a budget of
          0 an answer holds the tiles of the decode that made it.
          `PhotographReadError` gains `Changed`, `NotFound`, `TooLarge` and
          `Unreadable`, with `From<TileError>` and `From<GetError>`, and every
          caller reports them as its failure; `sfm web-export` reports the
          `RuntimeError` of a failed read as its failure. No function's
          signature changes, since 13e4a gave every read its caller's
          `Progress`; only `PhotographReadError` gains its variants. Tests: a
          kernel's `get_many` over 20 in-core photographs decodes them all into
          tiles in its read phase, so its renders decode nothing when the tiles
          fit the budget; a `get_many` over 20 photographs holds no pyramid (the
          cache's stats show only tiles, within a small budget); a read of a
          photograph whose file was removed after its tiles were evicted fails
          with `NotFound` naming it; `sfm web-export` given a `write_web_export`
          that raises `RuntimeError` reports it as its failure; a render
          cancelled while it decodes an evicted tile stops with the
          cancellation, and so does a bulk kernel cancelled while a nested call
          it hands `progress.silenced()` decodes one; under a budget of 0 two
          renders from one answer decode its photograph once; and the three
          tests step 13e3 removed come back, each with a photograph whose file
          was removed after its tiles were evicted: the display bitmaps' render,
          the viewer's open and its conversion to embedded patches each fail
          with the read's error, naming it, the conversion pushing no version.

          As built, the cache is a handle on its state, which sits in an
          `Arc` (`Shared`), and each photograph `get` answers holds another
          handle, with the path, the slot it was answered from and the
          layout of its decode (the crate-private `CachedTiles`). `get` and
          `get_many` keep `&self`, so the viewer's `Arc<PhotographCache>`, the
          web export's cache on its stack and the examples call them as
          before, and an answer read after its cache is dropped still reads,
          keeping the cache's tiles alive until it is dropped; `get` taking
          `self: &Arc<Self>` would have changed every caller's signature, and
          a `Weak` handle would have made such a read fail. A read takes the
          tiles the slot keeps under its lock, with no stat of the file and
          counted as no hit; for a tile not kept it stats the file, fails with
          `Changed` when the file's length or modification time is not the
          slot's, and otherwise asks the cache for the keys as `get_tile`
          does, decoding at most once, failing with `Changed` too when the
          decode's layout is not the recorded one. A read of several tiles
          asks for them together: `region`, and `copy_regions`, which asks
          for the tiles of every level a view-region pyramid copies in one
          request, so a render decodes at most once; the tile path and
          `source_clipped_share` read a tile at a time, through the slot. A
          `get` of a photograph some of whose tiles are kept answers at once,
          refreshing their place in the eviction order; one none of whose
          tiles is kept decodes, keeping every tile as unused and none of
          them evicted by its own store. `peek` and `peek_state` still answer
          a photograph made of every tile, since the GUI thread must never
          decode. Under a budget of 0, and for a decode whose slot left the
          map while it ran, the answer holds every tile of its decode.
          `PhotographReadError::Unreadable`'s path is an `Option`, `None`
          when it comes from `GetError::Unreadable`, whose message alone
          names the file, as `OutOfRange`'s is for a photograph made in
          memory; a read through a photograph `get` answered keeps the path.
          `get_level` and `view_region_pyramid` answer a failed read of their
          photograph as `TileError::Unreadable` (`OutOfRange` and `Cancelled`
          as before). `sfm web-export` reports the `RuntimeError` as a
          `click.ClickException` (exit status 1), not as the usage error of
          a refusal, since nothing in the command line caused it.

          Memory, in a release build on `dino_dog_toy` (85 photographs at
          2040 x 1536): a `get_many` over the 85 under a 50 MB budget held
          1,125 MB in its answers after the call and peaked 1,187 MB over its
          start before this step, and holds 77 MB and peaks 606 MB (the
          decodes in progress under the default build budget) after it; under
          a 4 GB budget the cache holds every tile, 1,074 MB, either way. The
          display bitmaps of that reconstruction under the 50 MB budget peak
          566 MB over the start rather than 1,195 MB, but decode 526 times
          rather than 85 and take 1.50 s rather than 0.28 s. The renders
          already take their points by reference image (step 11c); the extra
          decodes come from a render's decode keeping only the tiles it asked
          for as used, the rest of the decode going before the used tiles of
          photographs the kernel had finished with, and from 32 workers
          being in as many reference images at once. Step 13e5 keeps such a
          decode whole while the photograph is under half the budget. Under
          the default budget and under a budget of 0 the decodes, the time
          and the peak are as before. A single `get` of an 8000 x 6000 JPEG
          raises the process's peak private bytes over its private bytes before
          the call by 2.24 times the image's bytes, 2.25 times before this step
          by the same measure, and the cache keeps 1.34 times. No result
          changes: the display bitmaps and their reference observations, at
          a budget of 4 GB, 0 and 50 MB, on
          `seoul_bull_sculpture_ground_truth_v02`,
          `kerry_park_ground_truth_v02` and `dino_dog_toy_ground_truth_v01`,
          are the same bit for bit as before this step. Specified in
          [core/camera/photograph-cache.md](../core/camera/photograph-cache.md)
          § "Reads through the cache", with the tests in
          [photograph_cache/read_tests.rs](../../crates/sfmtool-core/src/camera/photograph_cache/read_tests.rs),
          [patch/spawn/tests.rs](../../crates/sfmtool-core/src/patch/spawn/tests.rs)
          (the spawn's localization, which it hands `progress.silenced()`,
          decoding a photograph whose tiles were evicted),
          [patch/display_bitmaps/tests.rs](../../crates/sfmtool-core/src/patch/display_bitmaps/tests.rs),
          [state/open/tests.rs](../../crates/sfm-explorer/src/state/open/tests.rs),
          [state/edits/tests.rs](../../crates/sfm-explorer/src/state/edits/tests.rs)
          (the files removed as the render phase starts, under a cache that
          keeps only the photograph it decoded last) and
          [tests/test_web_export.py](../../tests/test_web_export.py).
      - **13e5, the bulk kernels under a small budget.** **Built**, so step
        13e, and with it step 13, is complete. The bulk kernels' point order
        was already built, in step 11c (`BulkPointOrder`: by reference image,
        then the reference keypoint's row and `x`). What 13e5 changes is what
        a render's decode keeps (the companion's open question 12): a read
        through a photograph `get` or `get_many` answered, which decodes the
        photograph again for a tile it lost, keeps the tiles it asked for as
        used and the rest as unused, as before, and, while the photograph's
        tiles of every level are under half the cache's budget
        (`READ_SPARING_SHARE`), its eviction drops none of the photograph's
        tiles, as a `get`'s keeps its photograph whole until the next decode;
        over half, it is evicted as a tile request's is. A bulk kernel that
        reads each reference image's tiles one after another then finds the
        rest of the decode for its next renders, where before the rest went
        before the used tiles of photographs the kernel had finished with;
        under a budget of about one photograph or less, keeping one view's
        decode whole would push out the tiles the point's other views need.
        Nothing a kernel answers changes.

        Measured in a release build on `dino_dog_toy_ground_truth_v01`
        (85 photographs at 2040 x 1536, about 12.6 MB of tiles each), on
        32 threads unless stated, before and after this step, with the
        answers the same bit for bit (the display bitmaps and their
        reference observations, and every point's position and normal after
        *Refine all normals* over every tenth point and after *Refit all
        points*): the display bitmaps under 50 MB decode 539 to 556 times in
        1.39 to 1.54 s before and 259 to 288 times in 0.78 to 0.89 s after,
        and on one thread 429 times in 8.3 s before and 160 times in 2.9 s
        after (85 in the read phase and 75 in the renders, within the 170 of
        once in the read phase and once per reference image); under 4 GB and
        under 0 they decode 85 times in 0.21 s and 0.27 s, as before. The
        normals decode 618 to 658 times in 2.0 to 2.6 s under 600 MB before
        and 559 in 1.9 s after, 2,379 to 2,518 in 6.0 to 6.3 s under 300 MB
        before and 1,734 in 4.5 s after, and 6,745 in 14.8 s under 50 MB
        before and 4,656 in 10.5 s after. The refit decodes 1,898 to 2,103
        times in 5.6 to 6.0 s under 600 MB before and 1,704 in 5.1 s after,
        and 63,160 in 131 s under 100 MB before and 41,691 in 88 s after.
        The process's peak private bytes are as before: 528 to 557 MB for
        the display bitmaps under 50 MB (556 to 588 MB before), 934 MB and
        820 MB for the normals under 600 MB and 300 MB (927 MB and 827 MB
        before), and 949 MB for the refit under 600 MB (965 MB before). On
        the six photographs of the tests under a budget of one photograph,
        the refit's renders decode 133 times, as before, where keeping every
        decode whole decoded 554 times.

        Specified in
        [core/camera/photograph-cache.md](../core/camera/photograph-cache.md)
        § "Budget and eviction", with the tests in
        [bench/tests/bulk_budget.rs](../../crates/sfmtool-core/src/bench/tests/bulk_budget.rs),
        on six 128 x 128 photographs in a row written to a temporary
        directory, cut into 16 px tiles, nine points to each reference image,
        each seen by its reference image and that image's neighbours, read on
        a rayon pool of one thread: *Refit all points* under a budget of one
        photograph and under one smaller than a photograph's level 0
        completes with the answer it gives when every tile fits, the cache
        counting every decode; under a budget of four photographs, three of
        which one reference image's points are seen in, each photograph is
        decoded once in the read phase and at most once more for each
        reference image whose points it sees, and the same points by index,
        which take the reference images in turn, decode more; and the stored
        bitmaps' whole-cloud render under a budget of two and a half
        photographs decodes each photograph at most once in its renders,
        while the refit under a budget of one photograph decodes fewer times
        than there are observations. The budget of the point-order test
        holds one photograph more than one reference image's points are seen
        in, since the unused tiles of the photographs the current reference
        image's points still read, kept from their earlier decodes, go before
        the used tiles of the photograph the previous reference image's
        points read and the current one's do not. The three
        take 0.32 s (the run that completes), 0.14 s (the point order) and
        0.13 s (the decode kept whole).
14. **The out-of-core tier, off.** Photographs of a workspace with
    `mipmap_options` read through their mip level files: tiles from the
    files, mipmaps made in the slot when missing or stale (the misses of a
    `get_many` in groups sized by the build budget), `TileError::Changed`,
    and panels queueing photographs whose mipmap is not current for the
    *Build mipmaps* quiet task (its shared list and the job that drains it;
    step 15 adds the rest of the task). Tests: with the small threshold, a
    `get` on a photograph with no mipmap makes it and returns its pixels;
    two threads asking for
    one path make one mipmap; a `get_many` over 20 out-of-core photographs
    never has more builds in progress than the build budget allows, with a
    budget set small for the test and checked through the count; a panel
    that needs a photograph without a current mipmap adds it to the front of
    *Build mipmaps*' list once and draws the spinner, twenty such panels add
    twenty photographs to the one task's list, and no rayon task a panel
    starts makes or waits on a mipmap; a kernel in another task makes a
    missing mipmap on its own job thread, and waits on the slot when *Build
    mipmaps* is making it; a caller cancelled mid-build stops within one
    stage; a build reports its phases through the caller's `Progress`; no
    GUI-thread reader requests a tile of a photograph that is `Building`, and
    none calls `get` for an out-of-core photograph; a file replaced by one
    with another record during a read fails with `Changed`, and one made
    stale during a kernel fails it with the named error.

    The step is seven pull requests, 14a to 14e2b. The cache first learns each
    photograph's workspace setting, reading every photograph as before; then
    it reads an out-of-core photograph whose mipmap is current from its
    files; then `get` makes a missing or stale mipmap in the slot; then
    `get_many` makes its misses in groups sized by the build budget and
    reports what it made; then the panels act on the `Building` state and
    queue photographs for *Build mipmaps*, whose job makes them (14e1); then
    Image Detail and the 3D background draw the coarsest level as it is made
    (14e2a); and last the other panels' spinner and Image Detail's message
    for a workspace file error (14e2b). Reading
    comes before making, so that making a mipmap only hands the slot the
    files the reading part already reads, as `build_mipmaps` returns them.
    In a workspace without `mipmap_options`, which until step 17 is every
    workspace no one has edited by hand, every result and every picture is
    the same after each part. In one with the object, 14b changes the pixels
    of an out-of-core photograph whose mipmap is current to its mipmap's;
    between 14b and 14c one whose files are missing or stale is still read
    in core, from a whole decode of the original, and from 14c no kernel
    reads one so. A panel never makes a mipmap: until 14e1 its requests read
    an out-of-core photograph without a current mipmap whole in core, as
    they read every photograph today, and from 14e1 they wait on its
    `Building` state instead.

    - **14a, the workspace's setting in the cache.** **Built.** The cache
      finds each photograph's workspace root when it makes the photograph's
      slot, and reads that root's `mipmap_options` with
      `read_mipmap_options` once, keeping it by root, and again when the
      workspace file's length or modification time changes. A slot records
      the setting it was made under, and `get`, `get_many` and the tile
      requests make a slot again when the setting has changed since, as they
      do when the photograph's file has changed. While the object is invalid
      (or the workspace file cannot be read or parsed), every photograph of
      that workspace is refused with the new `GetError::Workspace`, which
      names the file and the key and is remembered as a failure is until the
      file changes; `request_layout` answers it as the layout's failure, the
      tile requests as `TileError::Unreadable`, and a read through a
      photograph the cache answered as `PhotographReadError::Unreadable`.
      Every photograph is still read in core, so nothing else changes.
      Signature changes: `GetError` gains `Workspace(WorkspaceConfigError)`.
      Tests: a photograph in a workspace whose object is invalid is refused
      by `get`, `get_many`, `request_layout`, `try_decode_tiles` and
      `get_tile` with the error naming the key, nothing decoded, with a
      budget and with a budget of 0; once the file is fixed the next `get`
      decodes it, and a photograph decoded before the file broke is refused
      after it; the object is read once for the photographs of one root,
      and once more after the file changes.
    - **14b, reading a current mipmap.** **Built.** The cache reads a
      photograph's layout under its setting (`photograph_layout`), and for
      an out-of-core photograph checks the coarsest mip level file when it
      first needs the photograph and each finer one when it first opens it,
      with one
      `MipmapSource` kept per original and spelling of its path for as long
      as the cache lives; it takes the region answer from the coarsest file's
      record while test 4 holds. An out-of-core photograph whose mipmap is
      current is `Known` with its out-of-core layout, and its tiles come
      from the files: levels 1 to `K` by `MipLevelFile::decode_region`, the
      levels below `K` box-filtered from a whole decode of level `K` and
      kept with it, and level 0 by `OriginalRegions` (indexed when its first
      level-0 tile is asked for, a failure of the index making the mipmap
      stale) or from `-mip0`. Each opened file's bytes and restart index
      are an entry under the one budget, and the coarsest level with the
      levels below it goes only after every other entry. A file whose bytes
      were evicted is read again and its record compared with the one
      recorded: a different record fails the read with the new
      `TileError::Changed` (and `PhotographReadError::Changed` through a
      photograph `get` answered), naming the file. An out-of-core photograph
      whose files are missing or stale is still read in core, as before.
      `request_layout` and `layout` answer the layout the photograph's tiles
      are served under: the out-of-core layout while its mipmap is current,
      the in-core one otherwise. The two have the same levels, level sizes
      and tile grid (they differ only in the tier and `K`), so a key valid
      under one is valid under the other, and a panel holding either is
      never answered `OutOfRange` for the change.
      **The panels until 14e.** A panel's request, `try_decode_tiles`, never
      answers `Changed`: a request that finds the photograph's mipmap not
      current, whether missing, stale or changed since the slot read it,
      marks the slot's mipmap not current and serves the tiles it was asked
      for from a whole decode in core, in the same request, as `get` serves
      a photograph whose files are missing. Under a budget of 0, where no
      slot remembers anything, each request decides from the files as they
      are and serves its tiles the same way, so no request is answered with
      an error a later one would answer again. The tiles a panel already
      holds from the files stay drawn beside the new ones until it asks for
      them again; both are the same photograph, a JPEG loss apart. The
      viewer's `failed_with` handles `TileError::Changed`, which only a
      kernel's read can now return, as it handles `Unreadable`: the
      photograph's kept answers are dropped and the failure kept with the
      photograph's file stamp, asked about again only when that stamp
      changes, never with `forget`, so a `Changed` cannot start a request
      and repaint loop. From 14e a `Changed` puts the photograph back to
      `Building` instead.
      Signature changes: `TileError` gains `Changed { path }`;
      `PhotographCacheStats` counts the file entries beside the tiles.
      Tests: every tile, region and level of an out-of-core photograph with
      a current mipmap equals the same pixels of its files (`decode_region`,
      `OriginalRegions::decode_region`, and the levels below `K` from level
      `K`), for a region-readable original and one with a `-mip0`; the
      coarsest level outlives every other entry under eviction; a file
      replaced by one with another record during a read fails with
      `Changed`, and one made stale during a kernel fails it with the named
      error; an in-core photograph in the same workspace reads as before;
      `request_layout` answers the in-core layout for an out-of-core
      photograph without a current mipmap and the out-of-core one with it;
      and Image Detail, Track View and the 3D background asking over many
      looks, through the viewer's `PhotographRequests`, for an out-of-core
      photograph whose mipmap is missing, and for one whose files are
      replaced with another record after its first tiles were answered,
      under a budget of 0 and under a budget smaller than the photograph,
      draw its tiles, start a bounded number of tasks and repaints (no more
      than the tiles they ask for need, however many looks), answer no
      `Changed`, and make no mipmap (the mipmap directory unchanged).
      **As built** ([photograph_cache/files.rs](../../crates/sfmtool-core/src/camera/photograph_cache/files.rs)):
      a slot decides its tier the first time a request needs the photograph
      (`get`, `get_many`, a tile request, `request_layout`), and only for a
      photograph its setting puts out of core, so a workspace without
      `mipmap_options` reads exactly as before, with no extra header read;
      a photograph too thin to store a reduced level (`K` of 0) is read in
      core. The cache finds the layout from the photograph's header and the
      setting it keeps, not with `photograph_layout`, and cuts level `K`'s
      own tiles from the same whole decode of its file as the levels below
      it, which gives the pixels `decode_region` would. A file found
      missing, not current or changed during a kernel's read marks the
      slot's mipmap not current as a panel's does, so the kernel fails with
      `Changed` and the photograph's next requests read it in core, as one
      whose files are stale is read until 14c. Under a
      budget of 0 `request_layout` decides from the coarsest file as it is,
      so it can answer the out-of-core layout for a photograph a panel then
      reads in core because a finer file is not current; the two layouts
      have the same grid. `get_many` still reserves a whole decode's
      estimate for each path it does not find made (14d resolves layouts
      first). `PhotographCacheStats` gained `files`, and its `bytes` count
      them. Measured with a counting allocator on a synthetic 8000 x 6000
      region-readable JPEG (8.4 MB, its mipmap made first), reading eight
      tiles of levels 0 and 1 with `get_tile`: out of core the reads peaked
      at 22.4 MB over their start and held 20.2 MB after (the eight tiles,
      6.3 MB, and the original's, level 1's and the coarsest file's
      entries); read in core, as in a workspace without the object, they
      peaked at 289 MB and held 194 MB (every tile of the whole decode, under
      a 4 GiB budget).
    - **14c, making a mipmap in the slot.** **Built.** A `get` that finds an
      out-of-core photograph's files missing or stale makes the mipmap in
      the path's slot on its own thread, reserving the build with
      `reserve_builds`, reporting the phases *decode*, *reduce* and *encode*
      through its `Progress` and polling it between stages; a cancelled
      maker leaves the slot empty for a waiter, and the next maker makes only
      the levels whose files are not current. The slot then reads from the
      files the build returned, and a build whose files could not be written
      is held as the in-memory equivalent, its bytes counted against the
      budget, with a warning in the log; a photograph that the hash finds is
      not a shared file's source makes its own mipmap over it, with a
      warning naming both paths. `get_tile`, `view_region_pyramid`,
      `get_level` and `try_decode_tiles` never make one: the first three,
      a kernel's reads, answer `Changed` for an out-of-core photograph whose
      files are not current, and `try_decode_tiles`, a panel's, serves such
      a photograph in core as in 14b until 14e. No kernel reads an
      out-of-core photograph in core from here on. Signature changes:
      `PhotographCacheStats`
      gains `mipmaps_made`; a build that fails is `GetError::Unreadable`
      naming the photograph. Tests: with the small threshold, a `get` on a
      photograph with no mipmap makes it and returns its pixels; two threads
      asking for one path make one mipmap; a caller cancelled mid-build
      stops within one stage, and the waiter that takes the build over makes
      only the levels not written; a build reports its phases through the
      caller's `Progress`; a build behind a blocked mipmap directory is read
      from memory and counted against the budget; and a panel's
      `try_decode_tiles` of an out-of-core photograph with no mipmap, with a
      budget and with a budget of 0, answers its tiles from a decode in core
      and makes no mipmap.
      **As built** ([photograph_cache/build.rs](../../crates/sfmtool-core/src/camera/photograph_cache/build.rs)):
      a slot's tier tells an out-of-core photograph whose mipmap is not
      current apart from an in-core one, so `get` and `get_many` make the
      mipmap there and `get_tile`, `view_region_pyramid`, `get_level` and a
      read through a photograph answer `Changed` naming the photograph (or
      the failure the slot remembers), decoding nothing; a panel's
      `try_decode_tiles` answers `Busy` while a build holds the slot.
      `get_many` already made a member's build through `get`'s path, so a
      member makes it too, in its group's share; until 14d it passes on only
      the build's messages through its `Progress`, since the phases of
      members running in parallel would overwrite one another. The maker
      checks the coarsest file again once it has marked the slot, and reads
      the files without a build when another process or an earlier build
      made them current since. "Makes only the levels not written" is
      `build_mipmaps` reading each level whose file is current rather than
      encoding it again, which the cache's own builds and every other
      caller share; and `build_mipmaps` writes the coarsest file last, after
      `-mip0`, still encoding it first. Without that, a maker cancelled
      after writing the coarsest file would leave a mipmap whose coarsest
      file is current and whose finer files are missing, which a new slot,
      checking the coarsest file alone as 14b does, takes for current, so a
      kernel would fail with `Changed` on its first finer read; since
      *Build mipmaps* gives way to every task asked for (14e), such
      leftovers would be common. The slot keeps the files the build returned
      as its file entries, with their records, and drops the tiles a panel's
      decode in core kept for it. A build held in memory is kept the same
      way, as file entries counted against the budget, evicted after every
      entry but the coarse tiles and spared by the slot's own eviction while
      under half the budget, as 13e5 spares a read's decode; a kernel's read
      of a level evicted since makes the build again in the slot, deterministic
      so its records match, keeping every level, and a panel's read of one
      decodes in core, kept by no slot, since a panel never makes a mipmap.
      A slot outside the map holds the levels it was made with, as an answer
      under a budget of 0 holds its decode, and the warning is sent once for
      each photograph. So in a read-only workspace the bytes kept pass the
      budget by at most one photograph's levels rather than pinning every
      photograph read (an audit finding; option (a), which the draft's
      "counted against the budget" asks for). Under a budget
      of 0 nothing is kept, so a `get` whose files could not be written
      builds again on its next call. `PhotographCacheStats::decodes` does
      not count a build, which `mipmaps_made` counts once it finishes;
      a build started is a miss. The warning for a file made over another
      spelling's is `build_mipmaps`'s, sent through the `Progress` it is
      given. Measured with a counting allocator in a release build on a
      synthetic 8000 x 6000 JPEG at 4:2:0 (10 MB; 144 MB of pixels), with a
      new cache for each call: a `get` that made the mipmap of a
      region-readable original peaked at 243 MB over its start (1.69 times
      the image's bytes, 5.1 bytes a pixel, about 1% over the 5 bytes a pixel
      the build reserves) and held 5.3 MB after (the files it wrote, kept as
      entries), in 0.48 s; one that also made a `-mip0` peaked at 244 MB and
      held 22.5 MB, in 0.90 s; a `get` of the same photograph whose mipmap
      was current peaked at 0.1 MB; and a whole decode in core, as in a
      workspace without the object, peaked at 289 MB (2.01 times, the same
      reservation) and held 194 MB. Encoding the full-resolution `-mip0`
      raised the build's peak by about 1 MB, so the peak comes before the
      encodes, while the original is decoded.
    - **14d, `get_many`'s builds.** **Built.** `get_many` first resolves
      each path's layout (a header read, the region test and a record check), so an
      out-of-core photograph whose mipmap is current is read without a
      build's reservation, and makes the misses (out-of-core photographs to
      build and in-core photographs none of whose tiles is kept) in groups
      sized by the build budget, counting the mipmaps it made beside the
      photographs it read; the callers' phase notes say so (*3 of 17 read,
      2 mipmaps made*). Signature changes: `GetManyTally` gains
      `mipmaps_made`. Tests: a `get_many` over 20 out-of-core photographs
      never has more builds in progress than the build budget allows, with
      a budget set small for the test and checked through the count; a
      `get_many` that finds a photograph's build in progress by another
      caller waits on the slot rather than start a second; its tally and
      its `Progress` count the mipmaps made.
      **As built** ([photograph_cache.rs](../../crates/sfmtool-core/src/camera/photograph_cache.rs)
      `get_many` and `look`): the look at each path, in parallel and
      reserving nothing, answers a path the cache answers as it is now
      (`made_now`, as before), and otherwise decides the slot's tier as a
      first request does (14b's `tier_of`, or under a budget of 0 the
      files as they are) and answers an out-of-core photograph whose
      coarsest file is current from its files; every other path is a miss
      sized from its header, an in-core decode and a build reserving the
      same whole-decode estimate, as 13b and 14c set. `GetManyTally` also
      gains `from_mipmaps`, the out-of-core photographs the call found
      current and read from their files, so a read from files is counted
      neither as `read` (decoded whole) nor as `reused`, which keeps its
      meaning of nothing read: before 14d such a photograph was `reused`
      with a budget and `read` under a budget of 0. The five counts add up
      to the paths asked for; a photograph whose slot already read from its
      files is `reused`, and one whose build another caller made while this
      call waited is `reused` too. The mipmaps made reach the `Progress` as
      the value `mipmaps_made` (`Progress::value`), the builds finished so
      far, as each finishes, since `get_many` already counts `images` and
      a stage shows its newest count. The notes add `, {f} read from mip
      level files` and `, {m} mipmaps made` (the bench's `decode images`,
      the cluster-patches build's `read photographs`) or `, {m} mipmaps
      made` (the bitmap render's `decode photographs`, *{k} of {n} read, 2
      mipmaps made*) only when the count is not 0, so a workspace without
      `mipmap_options` reads exactly as before. The test of builds in
      progress reads, as each build starts, the build count and then the
      builds started less those made, and asserts the count covers every
      build in progress and never passes the budget; a count read alone
      would not catch groups larger than their grant, since a member past
      the grant adds nothing to it. The count is read first because a build
      counts itself made before it gives its share back. The in-core test
      of 13b's groups reads its decodes in progress the same way, through
      the same helper. Measured with a counting allocator in a release
      build, a `get_many` over 12 synthetic 4000 x 3000 region-readable
      JPEGs (60 MB a build reserved) with no mipmap, under a 64 MiB budget,
      peaked at 167, 253 and 451 MB over its start with build budgets of
      one, two and four builds, holding 62 MB after, against 167, 272 and
      444 MB before 14d: 14c's groups already bounded the builds, so the
      peak is unchanged and grows with the build budget, about 95 MB a build
      in flight. Before 14d that call's tally said 12 read and no mipmap
      made. With six of the twelve made current first, the call reserved
      nothing for those six and took 2.5 s against 3.0 s before, peaking at
      266 MB against 272 MB.
    - **14e, the panels.** `PeekedLayout::Building { coarsest }` and
      `PeekedTile::Building`, `request_layout` answering `Building` when its
      record check finds the mipmap not current, a build
      publishing each reduced level as it is encoded, `display_layout`
      queueing a photograph that is `Building` at the front of *Build
      mipmaps*' shared list once per path, the spinner with *Building
      mipmap…*, the 3D background drawing the coarsest level while it
      waits, and Image Detail's *Workspace file error:* with the message.
      It looks again at how a panel is answered for an evicted level held
      in memory (§ "Open questions", 15). A panel's request then no
      longer reads an out-of-core photograph
      without a current mipmap in core, and a `Changed` puts the photograph
      back to `Building`, queued again. With the list comes the least of
      the *Build mipmaps* quiet task that drains it (open question 14,
      decided (a)): a quiet source on the fast queue, `state/build_mipmaps.rs`,
      whose job takes one chunk from the front of the list, as many
      photographs as `reserve_builds` would grant from an empty count and
      at most the rayon pool's size, calls `get_many` on it with its own
      `Progress` and ends, the source queueing it again while the list is
      not empty; it pushes no version, and the Background Tasks panel shows
      it as *Building mipmaps: {done} of {total}*. The viewer cancels every
      quiet task running in the queue a task someone asked for is queued in
      (`AppState::give_way_to`), so the job gives way from the first: a
      cancelled job puts the photographs of its chunk it had not finished
      back at the front of the list, and its source queues it again behind
      the displacing task, so no photograph a panel queued is lost. The
      source has its place in the `QuietRank` order, after the camera model
      proposal's job and before the evaluations, and does not wait for its
      node (`waits_for_its_node` answers `false`), so it runs on a locked
      node, since it reads photographs and not the node's version. Step 15
      adds the rest of the task. Tests: a panel that needs a photograph
      without a current mipmap adds it to the front of *Build mipmaps*'
      list once and draws the spinner, twenty such panels add twenty
      photographs to the one task's list, and no rayon task a panel starts
      makes or waits on a mipmap; no GUI-thread reader requests a tile of a
      photograph that is `Building`, and none calls `get` for an out-of-core
      photograph; a kernel in another task makes a missing mipmap on its own
      job thread, and waits on the slot when *Build mipmaps* is making it;
      the task runs on the fast queue, makes one chunk a run, is queued
      again while its list is not empty, pushes no version and runs on a
      locked node; it gives way to an asked-for task, leaving no slot
      marked as being made and the photographs of its chunk at the front of
      its list; and a panel's spinner ends once the task has made its
      photograph's mipmap.

      The step is three pull requests, split so that no part leaves a panel
      waiting on work nothing does (the reason for open question 14) and no
      part can ask for a photograph and repaint without end. 14e1 gives the
      panels the `Building` state together with the task that ends it, so
      the moment a panel waits, something makes what it waits for; 14e2a
      and 14e2b change only what a waiting panel draws.
      - **14e1, the panels wait on *Build mipmaps*.** **Built.**
        `PeekedLayout::Building` and `PeekedTile::Building`; `layout`,
        `request_layout` and `peek_tile` answering them for an out-of-core
        photograph whose mipmap is not current; `try_decode_tiles` reading
        no out-of-core photograph in core, answering `Changed` instead and
        leaving it `Building`; the viewer's `DisplayLayout::Building`,
        `display_layout` queueing the photograph at the front of the shared
        list once per path, Image Detail's spinner with *Building mipmap…*,
        the other panels' loading state, and the 3D background keeping the
        previous photograph; and the *Build mipmaps* quiet task as the 14e
        entry above describes it. Signature changes: `PeekedLayout` and
        `PeekedTile` gain `Building`; `PhotographCache::builds_that_fit`;
        `AppState::start_quiet_task` takes `node: Option<ReconId>`. Tests:
        every test the 14e entry lists.
        **As built** ([state/build_mipmaps.rs](../../crates/sfm-explorer/src/state/build_mipmaps.rs),
        [photograph_cache/tiles.rs](../../crates/sfmtool-core/src/camera/photograph_cache/tiles.rs),
        [photograph_cache/files.rs](../../crates/sfmtool-core/src/camera/photograph_cache/files.rs)):
        `Building` is a unit variant; 14e2 gives it the coarsest level. A
        mapped slot whose tier is not current is `Building` to `layout`
        and its tiles to `peek_tile`, a remembered failure first; the
        layouts `request_layout` keeps (a `PeekedLayout` each now) become
        `Known` when a build or a check reads the photograph's files and
        `Building` when a read finds one not current, so under a budget of
        0, which keeps no slot, the spinner ends when the task's build ends.
        The task is queued on no node: the photographs on its list belong
        to whichever nodes the panels show, so `start_quiet_task` takes the
        node as `Option<ReconId>` and the task's quiet line, having no node
        to name, is `quiet` and its label, *Building mipmaps: {done} of
        {total}* (the Background Tasks panel's `show_quiet_on_no_node`).
        `done` counts the photographs answered, made or found current, since
        the list was last empty. A chunk is sized by
        `PhotographCache::builds_that_fit` of the photograph at the front,
        from its header. A cancelled run puts back the photographs of its
        chunk whose layout is not `Known`. A photograph the run could not
        make is kept as a failed decode of the panels' reads, with its
        file's stamp, which is what keeps a budget of 0 from asking for it
        again on every frame. Track View, the Track Table's crops and
        Measurement View's endpoint crops draw their loading state while
        the photograph is `Building`; 14e2 gives them the spinner and its
        text. Image Detail does not keep drawing the previous image while
        the new one is `Building`, since a build takes seconds: the
        spinner says what it waits on, and there is no picture to click
        (G13). Two loops the plan did not name came up and are closed in the
        cache. A read that finds a finer file not current (a deleted `-mip1`,
        a file rewritten by another process) made the photograph `Building`,
        and the task's `get_many` then took the coarsest file's word that
        the mipmap was current and made nothing, so the panel's next read
        found the same file and the photograph went round for ever; the
        cache now notes the file found not current (`files::Stale`, keyed by
        the original's length and modification time), and while the
        original is unchanged and no build has made the mipmap since, the
        check of the coarsest file answers not current, so the next `get`
        or `get_many` builds, rewriting each level whose file is not
        current. A missing or rewritten file is one a build mends, so the
        same file found missing or rewritten again (a mipmap directory
        deleted twice, step 15's makers writing a file twice) is a first
        find once more and is built again. Only the same file found not
        current again after that build, still there with the length and
        modification time it had when first found (`Stale::file_stamp`), is
        one the build left as it was because its record is current though
        its data does not decode; no build mends it, and the photograph is
        then remembered as unreadable, naming the file, until the original
        changes or `forget` or `clear` drops it (both also drop the
        `not_written` note below). An audit of 14e1 found the first form of
        this rule, which counted any second find of the same file, turning
        a mipmap directory deleted twice into an unreadable photograph for
        the session. And under a budget of 0, which keeps no build, a mipmap
        whose files could not be written was made again on every request,
        each time a panel asked for its tiles; the cache now notes such a
        photograph with the original's stamp (`Shared::not_written`, which
        replaces the set the warning was sent once for), and while the
        original is unchanged `request_layout` answers its in-core layout and
        `try_decode_tiles` decodes its tiles in core, kept by no slot, as
        open question 15 (a) answers a panel's read of an evicted level held
        in memory. A kernel's `get` after a read found a file not current
        now builds where it read the files before, so a test of a file
        rebuilt by another process counts one mipmap made, which encodes
        nothing over the current files but decodes the original. Image
        Detail's prefetch of the next animated image queues it too, since
        it asks through `display_layout`.
      - **14e2, what a waiting panel draws.** Two pull requests: the
        coarsest level drawn by the two panels that show a whole photograph
        (14e2a), then the other panels' spinner and Image Detail's message
        for a refused workspace file (14e2b), each changing only what a
        waiting panel draws.
        - **14e2a, the coarsest level while a mipmap is made.** **Built.**
          `PeekedLayout::Building { coarsest }`; a build publishing the
          coarsest level to its slot as it is encoded (`build_mipmaps`
          telling its caller each level, which the cache cuts into the
          coarse tiles a finished mipmap keeps), so `layout` answers it
          about a second after a build starts; Image Detail drawing it
          under the spinner and the 3D background drawing it as the
          background while the photograph is `Building`. Signature
          changes: `PeekedLayout::Building` gains `coarsest`;
          `build_mipmaps` gains `on_level`; `DisplayLayout::Building`
          carries the coarsest level. Tests: a build held after its
          coarsest level is encoded (the cache's slot hook) answers
          `Building` with that level, and Image Detail and the 3D
          background draw it before the build ends; under a budget of 0
          the coarsest is handed to the panel that asked, as a tile is; the
          level drawn while waiting takes clicks as the photograph does
          (open question 16); and neither panel asks for a tile or
          repaints more than the reads need, at a budget of 0 and at one
          smaller than the photograph.
          **As built** ([photograph_cache/build.rs](../../crates/sfmtool-core/src/camera/photograph_cache/build.rs),
          [image_detail/mod.rs](../../crates/sfm-explorer/src/image_detail/mod.rs),
          [scene_renderer/upload/bg_image.rs](../../crates/sfm-explorer/src/scene_renderer/upload/bg_image.rs)):
          `coarsest` is a `CoarsestLevel`, the photograph's out-of-core
          layout and one tile of level `K` and of each level below it, not
          the one tile the plan named: a panel needs the level-0 size to
          lay the level over the pixels it covers and fit its view, which
          the tile alone does not give, and with the levels below `K` cut
          too Image Detail draws the level at any zoom through its usual
          fallback, and the 3D background at level `K`. They are cut from
          the level-`K` file as the finished mipmap's coarse tiles are
          (`files::cut_coarse`, which both share), so Image Detail keeps
          them as the photograph's own coarse tiles once it is `Known`.
          Only level `K` is published; `on_level` tells every level, and
          the finer ones would serve no panel, which asks for no tile while
          the photograph is `Building`. The slot holds the level
          (`SlotData::coarsest`) until it reads the build's files or drops
          its entries, not counted against the budget; under a budget of
          0 the build puts it in the layout `request_layout` kept for the
          path, when a panel asked for one and it is `Building`, which is
          how it reaches that panel. A failed build's level is dropped and
          a cancelled one's kept until the next build publishes, since the
          photograph is queued again; a build made again in memory for a
          kernel publishes nothing. The repaint: a quiet task's progress
          wakes no one, so the build reports `COARSEST_PUBLISHED` through
          its `Progress` (a `get_many` member passing it on beside the
          messages) and *Build mipmaps*' run, tapping its `Progress`, calls
          the viewer's `wake` once a build, so the repaints are bounded by
          the photographs built, one each besides the run's end. Image
          Detail's coarsest frames take clicks, drags and menus as the
          finished photograph does, under G1 to G12, since the level is the
          selected photograph at low resolution (open question 16; the
          first form held them under G13, which the maintainer reversed),
          draw the view at the photograph's own size so the overlays and
          the pixels a gesture names line up, ask for no tile, and paint
          the spinner and *Building mipmap…* over everything, on a
          backing, as painted shapes that take no pointer. The background copies level `K` into its texture once
          and keeps it while the photograph's own level loads once it is
          `Known`. An audit-time flake in
          `a_file_a_read_found_missing_is_made_again_each_time` (two writes
          within one tick of the file system's clock sharing a modification
          time) is closed by giving each rewrite a time of its own.
        - **14e2b, the other panels and a workspace file error.**
          **Built**, so step 14e and step 14 are complete. Track
          View's pictures, the Track Table's crop cells and hover, and
          Measurement View's endpoint crops drawing the spinner with
          *Building mipmap…* in place of their loading state
          (`PeekedRead::Building`); and Image Detail's *Workspace file
          error:* with the message for a photograph refused by an invalid
          `mipmap_options` (`DisplayLayout::Unreadable` carrying the cache's
          message, or a variant for a workspace's refusal). Signature
          changes: `PeekedRead` gains `Building`. Tests: a Track View
          picture, a Track Table crop and a Measurement View crop of a
          `Building` photograph draw the spinner and *Building mipmap…* and
          ask for no tile; Image Detail on a photograph of a workspace whose
          `mipmap_options` is invalid draws *Workspace file error:* and the
          message naming the key; and none of these asks for a tile or
          repaints more than the reads need, at a budget of 0 and at one
          smaller than the photograph.
          **As built** ([state/photograph_requests.rs](../../crates/sfm-explorer/src/state/photograph_requests.rs),
          [patch_pictures.rs](../../crates/sfm-explorer/src/patch_pictures.rs)):
          a render through `read_peeked` or `read_peeked_without_asking` is
          `PeekedRead::Building` while the layout is `Building`; only the
          read that asks queues the photograph, so the Track Table's rows
          queue nothing while they scroll and do once they stop. The three
          panels share one painter, `patch_pictures::paint_building_mipmap`,
          which paints the spinner in the cell and *Building mipmap…* under
          it, wrapped to the cell's width, where the two fit, and the
          spinner alone where they do not (the Track Table's 16 pt
          pictures); each cell's hover says the label too, beside a
          spinner (`patch_pictures::show_building_mipmap`), as does a hover
          view whose photograph waits. Nothing is kept for a waiting
          picture: Track View notes its rows waiting on each frame
          (`TrackBody::building`), the Track Table's cell is
          `CellPicture::Building` from `Made::Building` and its hover
          `NotCut::Building`, and Measurement View's crop is
          `Crop::Building`, asked for again on every frame like an
          unreadable one and dropped on the frame its tiles start to load,
          so the cell then says *loading…* rather than keep the spinner.
          The spinner asks for the frames it turns in, so a panel repaints
          every frame only while it draws one, which the run of *Build
          mipmaps* ends. For the workspace refusal the viewer has the
          variant, `DisplayLayout::WorkspaceFileError` with the
          `WorkspaceConfigError`, which Image Detail draws as *Workspace
          file error:* over the error, wrapped to the panel; the other
          panels and the 3D background treat it as `Unreadable`. To tell a
          workspace's refusal from an unreadable photograph without reading
          the message, the cache's `PeekedLayout::Unreadable` and
          `TileError::Unreadable`'s `why` carry the `GetError` in place of
          its message, two signature changes the plan did not name; their
          messages are unchanged. A refusal is kept as a failure, from the
          header read's task, from a decode's, or from a build of *Build
          mipmaps* the file refused because it broke after the photograph
          was queued (`failed_to_build` reads the header again, since
          `get_many` answers `None` with no error and a cache of budget 0
          keeps none, so no signature changes), watching the workspace
          file's stamp rather than the photograph's. The header read, the
          decode and the build take that stamp before a second read that
          finds the refusal still there (`confirmed_refusal`), so a file
          fixed while the first read ran is not kept with the fixed file's
          stamp, which no check would find changed (an audit finding); a
          decode whose second read no longer finds the refusal keeps no
          failure, and the next frame asks again. Without watching
          that file, fixing
          the file would leave every photograph the viewer had found
          refused answered as refused for the session, since `layout`
          answers a refusal without a stat. When the check of a later look
          finds the file changed it drops the failure, has the cache
          `forget` the path and repaints, and the next frame reads the
          photograph's header again. Tests, each under a budget smaller
          than a photograph and one of 0: the requests answering a render
          `Building`, queued once by an asking read and not by a looking
          one, and the workspace refusal found by a header read and by a
          decode, decoded no more, repainted for no more, and read again
          once the file is fixed, and one queued whose workspace file
          breaks before its build runs answered the same way and read,
          queued and made once the file is fixed
          (photograph_requests/tests.rs); Image
          Detail's *Workspace file error:* with the message naming the file
          and the key, repainting nothing; and Track View's tile and crop
          cells, the Track Table's crops and Measurement View's endpoint
          crops drawing the spinner and the label, the first two on hover
          too, asking for no tile, queueing each photograph once, and
          reading their pictures with a few reads once *Build mipmaps* has
          made the mipmaps, Measurement View's then asking for no repaint.
          Track View, unlike Image Detail and Measurement View, asks for a
          repaint on every headless frame of its tests whatever its
          photographs' state, since its header turns a spinner while the
          track is evaluated and no evaluation ends in those tests, so its
          test bounds the reads and not the repaints.
15. **Makers.** **Built**, in 15a to 15e below: `sfm sift` and `build_sift_file`, the rest of the viewer's
    *Build mipmaps* quiet task, whose job, shared list, `QuietRank` place
    and giving way 14e builds (the list filled when a reconstruction opens,
    the selected image first, then the Image Browser strip's, then the
    rest), `sfm ws mipmaps` with `--prune`
    and `--check` (its `--enable` waits for step 17, since it writes the
    object), and the thumbnail from the coarsest mip level file.
    Tests: `sfm sift` on a small-threshold synthetic workspace writes the
    files; the quiet task makes the selected image's mipmap in its first
    run, takes a photograph a panel adds to the front of its list in its
    next run without cancelling the running one, lets a waiting quiet task
    of higher `QuietRank` (a camera model proposal's job) start between its
    chunks while it has a long list;
    `--prune` deletes an orphan and
    keeps a current mipmap, and deletes every mip level file in a workspace without the
    object; `--check` writes nothing.

    The step is five pull requests, 15a to 15e, each a maker or a part of
    one that works on its own: the Rust `.sift` build first, since it needs
    no binding and no decision; then the viewer's two parts, the list
    filled at open and the thumbnail; then the Python makers, which need a
    binding. In a workspace without `mipmap_options`, which until step 17
    is every workspace no one has edited by hand, every result, file and
    picture is the same after each part.

    - **15a, `build_sift_file` makes mipmaps.** **Built.** In a workspace
      whose file has `mipmap_options`, each image `build_sift_file` decodes
      that is out of core and whose mipmap is not current (`check_mipmap`
      against a `MipmapSource` read for it) gets its mipmap made by
      `build_mipmaps` with the decode as `decoded`, so the viewer's *Build
      SIFT files* and `pick-images`, in the viewer and on the command line,
      make the mipmaps of the photographs they extract. No signature
      changes. Tests: a build in a workspace with a threshold of 64 writes
      the mip level files of the 300 x 200 photograph and none for the
      60 x 40 one, reporting no phase of the build; a workspace without the
      object gets no mipmap directory; an invalid object is a warning
      naming each image, with every `.sift` still written.
      **As built** ([sift/files.rs](../../crates/sfmtool-core/src/features/sift/files.rs)):
      the build runs after the decode, the hash and the thumbnail and before
      the gray conversion, while the decode is held anyway, rather than after
      the `.sift` is written as the Python extractors will: kept until after
      the extraction, the decode would sit beside the extraction's own
      memory, and made before it, the build's few bytes per source pixel
      stay below the extraction's peak, so the number of images run at once
      is unchanged. A build that fails, an invalid object and files that
      could not be written are warnings through the caller's `Progress`, and
      the `.sift` file is still written, as `build_mipmaps`' report of
      failures in § "Python bindings" does not stop a run; a cancelled build
      is the run's cancellation. Only the build's messages reach the
      caller's `Progress`, since its phases and counts would replace
      `build_sift_files`' image count.
    - **15b, the list filled at open.** When a reconstruction opens in a
      workspace with the object, the open job finds its out-of-core
      photographs whose mipmaps are not current (a `stat`, the header, the
      region test's markers and the coarsest file's record each) and puts
      them on *Build mipmaps*' list, the selected image first, then the
      Image Browser strip's, then the rest in image order, behind any a
      panel has queued, and the task is queued. Tests: the task makes the
      selected image's mipmap in its first run; takes a photograph a panel
      adds to the front of its list in its next run without cancelling the
      running one; and lets a waiting quiet task of higher `QuietRank` (a
      camera model proposal's job) start between its chunks while it has a
      long list. **Built.**
      **As built** ([build_mipmaps.rs](../../crates/sfm-explorer/src/state/build_mipmaps.rs),
      [open.rs](../../crates/sfm-explorer/src/state/open.rs)): the open's
      worker, after its patch bitmaps (whose reads make the mipmaps they
      need), reads with `PhotographCache::request_layout` the layout of each
      photograph in a workspace whose file has `mipmap_options`, on the rayon
      pool, in a phase `mipmaps` noting `{k} of {n} photographs need their
      mipmaps made`, and `Loaded` carries those that are `Building`. Which
      workspaces have the object is found from each image folder's
      workspace file, read once, so in a workspace without it no
      photograph's header is read. The GUI thread puts them on the list as
      it appends the node (`AppState::queue_opened_mipmaps`), in image
      order. No image of a node that has just opened is selected (appending
      it selects the node, which clears the image selection), and the Image
      Browser strip shows it from its first image, so image order is the
      plan's order: the strip's images first, and the selected image first
      once there is one, since Image Detail queues it, which moves a path
      the open queued from further back to the front (`MipmapList` keeps
      the panels' part of the list and the opens' apart for this). A second
      open's photographs go ahead of an earlier open's leftovers, since its
      node is the one selected. Closing a node, or Close All, takes every
      photograph that no node left in the scene has off the list
      (`AppState::forget_closed_mipmaps`; § "Open questions" 19): no later
      run takes one that waits, and a cancelled run does not put back one
      of its chunk unless an open or a panel queued it again before the run
      ended, which adds no second copy, though a running chunk is left to
      finish; the quiet line counts such a photograph neither as done nor
      in the total. The plan did
      not name this: the 14e1 list kept a closed node's paths and went on
      making them, and with the list now holding a whole reconstruction that
      could be minutes of builds of photographs nothing on screen shows;
      `sfm ws mipmaps` (15e) is the way to prepare a workspace whose
      reconstruction is not open. The open's layout reads stay in the cache,
      so a panel that shows one of the photographs reads none of them again.
      Nothing here starves another task: a run still gives way to a task
      someone asks for in the fast queue, and since it ends after each
      chunk, the first waiting quiet task in `QuietRank`, whatever its rank,
      starts before the next chunk (the queue starts it when the run ends,
      before the next frame queues the next run). Tests
      (`state::build_mipmaps::tests`, in a temporary workspace with a 64 px
      threshold and 300 x 200 PNGs): the open queues the five out-of-core
      photographs without a mipmap, and not the in-core one, the one with a
      current mipmap or the missing one, and an image selected before the
      first run starts is that run's chunk; a photograph a panel queues
      while a run is held is the next run's chunk, the held run not asked to
      stop; a proposal's job queued while a run is held starts between that
      run and the next, with the list still long; closing one of two nodes
      of the same file keeps the list, and closing both or Close All empties
      it; a workspace without the object queues nothing and reads no layout;
      the list keeps the panels' part ahead of the opens', the newest open's
      first; and a photograph a running chunk took, dropped by a close and
      queued again by an open or a panel before the run ends, is on the list
      once whether the run makes it or is cancelled, one not queued again is
      not put back, and the quiet line counts a dropped one in neither number.
    - **15c, the thumbnail from the coarsest mip level file.** The display
      thumbnail of an out-of-core photograph whose coarsest mip level file
      is current is made from that file; one whose file is not current gets
      the placeholder row at open, filled on the rayon pool once the
      photograph is `Known` (§ "Readers on the GUI thread"). `sfm
      web-export`, which fills a missing column with the same rows, makes
      the mipmap of an out-of-core photograph whose mipmap is not current
      through the photograph cache it reads with, then reads the coarsest
      file, so its rows are the viewer's (open question 18). Tests: the row
      equals `thumbnail_from_rgb` of the coarsest level's decode with no
      whole decode of the original; a placeholder row is filled once the
      task has made the mipmap; a web export of an out-of-core photograph
      with no mipmap writes its mipmap and the same row. Visible: a
      thumbnail's pixels change slightly.

      Built in two parts, since the fill is a viewer change of its own
      (rows replaced in a node's column after the open, and the atlas and
      the Image Browser's textures made again for them):
      - **15c1, the row from the coarsest file.** **Built.**
        `display_thumbnail_row` takes the photograph cache, a
        `NotCurrentMipmap` (`Make` or `DecodeOriginal`) and a `Progress`,
        and answers `Cancelled`; an out-of-core photograph's row is
        `thumbnail_from_rgb` of level `K`, read whole by the cache's
        crate-private `coarsest_level`
        ([tiles.rs](../../crates/sfmtool-core/src/camera/photograph_cache/tiles.rs),
        [thumbnail.rs](../../crates/sfmtool-core/src/reconstruction/thumbnail.rs)).
        In a workspace without `mipmap_options` it reads no header and the
        row is a whole decode outside the cache, byte for byte today's.
        `sfm web-export` passes `Make` through the budget-0 cache its
        bitmap render reads with, now made once for both
        ([web_export.rs](../../crates/sfmtool-core/src/web_export.rs)). The
        viewer's open passes its own cache and `DecodeOriginal`
        ([display_thumbnails.rs](../../crates/sfm-explorer/src/display_thumbnails.rs)),
        so a photograph whose mipmap is current is read from the coarsest
        file and one whose mipmap is not is decoded whole, as before this
        step, until 15c2. Tests: the row of a photograph with a current
        mipmap is the coarsest file's under either choice and differs from
        the whole decode's, `DecodeOriginal` decodes a photograph without
        one whole and makes nothing, `Make` makes it and reads the row from
        it with no whole decode in the cache, and a cancelled build is
        `Cancelled` (`reconstruction::thumbnail::tests`); a workspace
        without the object decodes whole outside the cache and writes no
        mipmap directory; the open's build reads a current coarsest file
        and makes no mipmap (`display_thumbnails::tests`); and a web export
        makes every missing mipmap, a second export reading them writes the
        same thumbnail pages, and an export without the object writes other
        pages (`web_export::tests`).
      - **15c2, the placeholder and its fill.** **Built.** The open gives a
        photograph whose mipmap is not current the placeholder row instead
        of a whole decode, and the row is filled on the rayon pool from the
        coarsest file once *Build mipmaps* reports the photograph made.
        **As built**
        ([thumbnail.rs](../../crates/sfmtool-core/src/reconstruction/thumbnail.rs),
        [display_thumbnails.rs](../../crates/sfm-explorer/src/display_thumbnails.rs),
        [thumbnail_fills.rs](../../crates/sfm-explorer/src/state/thumbnail_fills.rs)):
        `NotCurrentMipmap` gains `Placeholder`, which the open passes, and
        `ThumbnailSource` gains `AwaitingMipmap`, the grey row it answers
        for an out-of-core photograph whose mipmap is not current
        (`TileError::Changed` from `coarsest_level`), reading the header and
        the record and decoding nothing; the photograph part of
        `display_thumbnail_row` is now `photograph_thumbnail_row`, taking a
        path, which the fill calls. `DisplayThumbnails` remembers the rows
        awaiting a mipmap by path, and `with_rows` gives a new column with
        rows filled in, sharing the built column and every row filled
        before, with a count of fills and the fill that changed each row
        (`fills`, `filled_after`). `MipmapList` records each photograph a
        run answers, and `AppState::drive_thumbnail_fills`, once a frame
        after `drive_build_mipmaps`, takes them, starts one read on the
        rayon pool for each a node's row awaits (once while it runs), waking
        the viewer as it lands, and replaces the column of each node whose
        row awaits a landed photograph; no version is pushed and nothing
        reaches a save. The frustum atlas keeps its texture and writes only
        the cells filled since it was written
        (`SceneRenderer::thumbnails_changed`, then `upload_thumbnails`), and
        the Image Browser drops those rows' textures, which load again, and
        writes those images' barcode columns again (`patch_barcode`, one
        `set_partial` a column) rather than building the whole barcode, which
        reads every row's pixels. A fill copies nothing of the column: the
        names, the awaiting rows and the filled rows are shared behind an
        `Arc`, each filled row set once, and a value sees only the fills up
        to its own number. Two cases the plan did not name: the fill reads with
        `NotCurrentMipmap::DecodeOriginal`, so a photograph the task answers
        whose mipmap is still not current (a failed build, or levels that
        could not be written and were not kept under a budget of 0) is
        decoded whole, the rule `display_thumbnail_row` already had for a
        mipmap that cannot be made or read, and is read once, with no
        request queued again; and a row whose mipmap the open's patch bitmap
        render made after the thumbnails stage is not on the list, so the
        open fills it itself, on its worker, after its `mipmaps` stage. A
        node closed before its rows land gets none of them. Tests: the core
        row of a photograph without a current mipmap under `Placeholder` is
        the grey `AwaitingMipmap`, with no decode and no mipmap
        (`reconstruction::thumbnail::tests`); the open leaves five rows
        awaiting, decoding none, and once the task has made their mipmaps
        five reads fill them from the coarsest files, the history, the value
        and the unsaved state unchanged and no further read, build or
        repaint following; under a budget of 0 with the level files
        unwritable each answered row is decoded whole once and nothing is
        queued again; reads that land after their node closed change
        nothing, and an answered photograph no row awaits starts no read;
        and the open fills the rows whose mipmaps its patch bitmaps made
        (`state::thumbnail_fills::tests`); `with_rows` replaces only awaited
        rows and shares the rest (`display_thumbnails::tests`); the atlas
        writes only the filled cells into the same texture
        (`scene_renderer::upload::tests`); and the Image Browser loads only
        the filled row's texture again, with its pixels, and writes only
        that image's barcode column, the barcode then equal to one built
        whole (`image_browser::tests`).
    - **15d, the Python makers and `sfm sift`.** The `sfmtool.photographs`
      submodule with the `PhotographCache` class, only its constructor,
      `build_mipmaps` and `stats` (open question 17), wired as § "Python
      bindings" says; the Python extractors calling `build_mipmaps` after
      each `.sift` is written, on one cache the run holds; and `sfm sift
      --extract` making the mipmaps of the photographs it extracts. Tests:
      `sfm sift` on a small-threshold synthetic workspace writes the files;
      in a workspace without the object it writes none; the new submodule
      passes `tests/test_module_layout.py`, the registration test and the
      lazy-loading test. **Built.**
      **As built** ([photographs.rs](../../crates/sfmtool-py/src/photographs.rs),
      [photograph_cache/build.rs](../../crates/sfmtool-core/src/camera/photograph_cache/build.rs),
      [sift/extract.py](../../src/sfmtool/sift/extract.py)): the binding's
      work is core's `PhotographCache::build_mipmaps`, which reads each
      path's header and workspace on the rayon pool and hands only the
      out-of-core photographs that store a level to one `get_many`, so no
      photograph in core is decoded, as a `get_many` over every path would
      decode it, and answers a `MipmapBuildReport` of the paths `made`,
      `current`, `in_core` and `failed` (with the `GetError`); `get_many`'s
      answers with how each was found are now a private
      `get_many_answers`, which `get_many` tallies. The binding runs the
      call on a thread of its own while the calling thread runs Python's
      signal handlers every 50 ms, as `refine_normals` does, so Ctrl+C
      stops a build within one stage, and sets its `ProgressCounter` from
      the value `mipmaps_made`. Two things differ from the plan.
      `image_files_to_sift_files` calls `build_mipmaps` once, after the
      `SiftWriteQueue` has written every `.sift` of the run, over the
      images it extracted, on a cache of budget 0, rather than after each
      file: a build per file would run beside the extraction of the next
      images, whose decodes are held then, and add a photograph's decode
      and reductions to the command's peak memory, while after the run the
      extraction's memory is freed and the builds, bounded by the default
      build budget (a quarter of memory) where the extraction runs as many
      images as fit in half, stay below it (measured on two 24-megapixel
      JPEGs: about 960 MiB at the peak with the object and without it, at
      one and at the default number of images at once); and the budget is
      0 since the run reads nothing back. The mipmaps of the photographs
      of all three feature tools are made this way, with a decode of their
      own (open question 4), the Rust `build_sift_file` keeping its own
      for the viewer and `pick-images`' `sfmtool` tool; `sfm solve`,
      `sfm match`, `sfm to-colmap-db` and `pick-images`' other tools,
      which extract through the same function, make them too. And 15d's
      `build_mipmaps` takes photographs' paths: the expansion of
      directories, whose rule `sfm ws mipmaps` sets, comes with that command
      in 15e. `sfm sift`'s summary gains the line `New mipmaps: {made} /
      {made + current} out-of-core image(s)` when there is an out-of-core
      photograph, and a `Warning: no mipmap was made: …` line on standard
      error for each failure, the `.sift` still written; without the object
      it prints nothing new, and the `.sift` files are the same bytes either
      way. Tests: `build_mipmaps` makes the missing mipmap, names each path
      in its list and decodes nothing whole, and a second call finds both
      current, with a budget and one of 0, a workspace without the object
      names the photograph in core and writes no mipmap directory, and a
      cancelled call makes nothing (`photograph_cache::build_tests`); the
      binding's report, counter, `stats` keys and budgets, and a negative
      budget's `ValueError` (`tests/rust_bindings/photographs/`); `sfm sift`
      writes the mip level files of the 300 x 200 photograph and none of
      the 60 x 40 one, and in a workspace without the object no mipmap
      directory and no word of mipmaps, with the same `.sift` bytes, and an
      invalid object warns of each photograph and writes every `.sift`
      (`tests/sift/test_sift_workspace.py`).
    - **15e, `sfm ws mipmaps PATHS…` with `--prune` and `--check`.** The
      `ws` subcommand, its spec beside `ws-init-command.md`, and the core
      functions that prune and check, behind a binding; the builds through
      15d's `PhotographCache.build_mipmaps`, which gains the expansion of a
      directory among its paths, in core, by the rule the command sets. Tests: `--prune` deletes an
      orphan and keeps a current mipmap, and deletes every mip level file in
      a workspace without the object; `--check` writes nothing. Visible:
      the new subcommand's help. **Built.**
      **As built** ([ws-mipmaps-command.md](../cli/workspace/ws-mipmaps-command.md),
      [mipmap/check.rs](../../crates/sfmtool-core/src/camera/mipmap/check.rs),
      [mipmap/prune.rs](../../crates/sfmtool-core/src/camera/mipmap/prune.rs),
      [photographs.rs](../../crates/sfmtool-py/src/photographs.rs),
      [_commands/ws.py](../../src/sfmtool/_commands/ws.py)): the core
      functions are `check_mipmaps(photographs, tile_size, progress)` and
      `prune_mipmaps(paths, &PruneMipmapsOptions { dry_run, tile_size },
      progress)`, bound as the free functions `check_mipmaps` and
      `prune_mipmaps` of `sfmtool.photographs`; § "Python bindings" makes
      `build_mipmaps` the one way to build from Python, and neither of these
      builds. The three modes are exclusive: no flag builds, `--check`
      reports and writes nothing, and `--prune` deletes orphans and makes no
      mipmap, so freeing the space of deleted photographs does not also wait
      for every missing build; `--check` lists the files from a dry run of
      the prune and exits with status 1 when anything is missing, stale or an
      orphan. Three things differ from the plan. The expansion of
      directories stays in Python: the command expands its paths with
      `expand_paths` and `SIFT_IMAGE_EXTENSIONS` (`.png`, `.jpg`, `.jpeg`),
      the rule and list `sfm sift` already used, now one named constant both
      read, and passes photographs to `build_mipmaps` and `check_mipmaps`,
      which keep taking photographs' paths; a rule in core as well would be a
      second implementation of the one `expand_paths` holds. A prune
      searches only the part of each workspace's mipmap directory that
      mirrors its paths, and the whole mipmap directory of each workspace
      nested below a directory path, rather than the whole directory of each
      workspace a path is in, as a build and a check reach only the
      photographs under the paths; a path need not exist, so a deleted
      photograph's or directory's path finds what it left. And the orphans
      are judged a file at a time against the plan's list plus the cases it
      did not name: a photograph now in core or too large, a level past the
      coarsest the mipmap stores, a `-mip0` beside a region-readable
      original, and a file in the enclosing workspace's directory for a
      photograph of a nested workspace; judged a file at a time, a stopped
      build's current finer files are kept for the next build. What a prune
      deletes is bounded as the user asked: only regular files inside a
      mipmap directory, named as a mip level file or a build's temporary
      file, each checked again just before its delete to be a regular file
      whose resolved directory is inside the resolved mipmap directory; no
      link or junction is entered or deleted, no directory is deleted, and a
      workspace whose `mipmap_options` is invalid is not searched. The
      mipmaps of photographs in no reconstruction are judged as any other
      and kept while current. Tests: the core cases in
      `camera::mipmap::check::tests` and `camera::mipmap::prune::tests`
      (every kind of orphan beside every kind of kept file, a dry run, a
      workspace without the object, the mirror of a path, a nested
      workspace, junctions and links, and the last check before a delete),
      the bindings in
      `tests/rust_bindings/photographs/test_mipmap_check_prune_rust_bindings.py`,
      and the command in `tests/test_ws_mipmaps.py`.
16. **Python and the readers that must agree with the viewer.** **Built**,
    in 16a to 16e below: the
    `PhotographCache` class's layout, level, tile and region reads, added to
    the submodule and class step 15d brings with the constructor,
    `build_mipmaps` and `stats` (open question 17), and core's
    `read_layout` and `read_region` behind it; `ImagePyramidSet.read` and
    `write_web_export` with `cache=`; and `embed-patches`, the
    strips, the `xform` steps, `sfm cluster-patches` and the `scripts/`
    harnesses on `ImagePyramidSet.read`, with its photographs raising
    `PhotographReadError` through the bindings.
    Tests: with the small threshold, the CLI and the viewer's code give the
    same results for the same photograph; in-core results unchanged; a
    binding whose file changes under it raises an error naming the path; a
    `PhotographCache` passed to two `ImagePyramidSet.read` calls makes and
    decodes the tiles once (its `stats` show the second call's reads as hits),
    while two calls without `cache=` share nothing; a level, a tile and a
    region read through the class equal what a kernel samples from the same
    file (a `Bilinear` render at level 0 and a `BilinearMip` render pinned to
    a level read back the same pixels); the arrays are read-only; a level,
    tile or rectangle outside the photograph, an empty rectangle, a negative
    index and a negative budget raise `ValueError`, a missing file
    `FileNotFoundError`, and an invalid `mipmap_options` `RuntimeError`; the
    class's new methods pass the registration test.

    Step 16 is five pull requests. Each moves a reader only once the
    binding it reads through is built, and a reader that has not moved
    keeps reading `read_image_rgb` or `read_workspace_image` arrays, whose
    pixels are the in-core ones, so no part leaves a reader without pixels.
    `sfm cluster-patches` moves in the second, since it is the reader whose
    memory the user reported (36 photographs of 200 MP, every one decoded
    whole at once).

    - **16a, the class's reads.** **Built.** Core's `read_layout`, with
      `PhotographLayout`, and `read_level`, `read_tile` and `read_region`
      in [photograph_cache/pixels.rs](../../crates/sfmtool-core/src/camera/photograph_cache/pixels.rs),
      each read the photograph `get` answers and copied from the tiles the
      rectangle covers, so it makes a missing mipmap first and reports
      `mipmaps_made`; and the Python `PhotographLayout` and
      `PhotographCache.layout`, `read_level`, `read_tile` and
      `read_region`. Two choices made in building it, against the
      interface above: core has a read for each of the three Python
      methods, rather than `read_region` alone, since `read_tile` and
      `read_level` need the layout `get` answers to know their rectangle
      and a binding that read the layout first would read the header
      twice; and the three answer `TileError`, as `get_tile` and
      `get_level` do, rather than `PhotographReadError`, since its
      `Unreadable` merges `GetError::Unreadable` and `GetError::Workspace`,
      which Python raises as `OSError` and `RuntimeError`.
      The Python `PhotographLayout` also has `tile_size`, the side
      `read_tile`'s columns and rows count in. No reader moves, so no
      result changes.
      Tests: in
      [pixels/tests.rs](../../crates/sfmtool-core/src/camera/photograph_cache/pixels/tests.rs),
      the layout `get` answers, a level, tiles and a region equal to a
      `Bilinear` render at level 0 and a `BilinearMip` render pinned to
      each level through another cache, the mipmap made once, and the
      errors; in
      [test_photograph_cache_rust_bindings.py](../../tests/rust_bindings/photographs/test_photograph_cache_rust_bindings.py),
      the layout's fields, read-only arrays equal to `read_image_rgb` in
      core and to the level's slice, the counter, and the Python
      exceptions; and the registration test's new names. Specified in
      [photograph-cache.md § "Layouts and pixels for a script"](../core/camera/photograph-cache.md#layouts-and-pixels-for-a-script)
      and [python-bindings.md](../python-bindings.md).
    - **16b, `ImagePyramidSet.read` and `sfm cluster-patches`.**
      **Built.** `ImagePyramidSet.read` in
      [views.rs](../../crates/sfmtool-py/src/patches/views.rs), over
      `get_photographs` in
      [photographs.rs](../../crates/sfmtool-py/src/photographs.rs); the
      cluster binding taking a set in
      [cluster.rs](../../crates/sfmtool-py/src/matching/cluster.rs), beside
      the list of arrays it still takes, as the `PatchCloud` kernels do; and
      `sfm cluster-patches` reading through it, with a cache it makes, and
      printing the cache's whole decodes, mipmaps made and budget. Choices
      made in building it: `TiledPhotograph` is `Clone`, a clone copying no
      pixel, so the set keeps the photographs `get_many` answers as the
      `Vec<TiledPhotograph>` every kernel binding already reads rather than
      a set of `Arc`s that would change them all; `read` checks a
      reconstruction's photographs against its cameras, as the constructor
      does, and refuses `workspace_dir` with a list of paths
      (`ValueError`); and a cache `read` makes for itself lives as long as
      the set, since the set's photographs read through it. In core the
      command writes the same file as before, byte for byte. The measured
      cost, and the decision on it, are open question 21.
      Tests: in
      [test_image_pyramid_set_read_rust_bindings.py](../../tests/rust_bindings/patches/test_image_pyramid_set_read_rust_bindings.py),
      a list of paths read in order and refined as the arrays of
      `read_image_rgb` are, a `CameraViews`, a string, `workspace_dir` with
      paths and a missing file refused, a reconstruction read under its
      workspace and checked against its cameras, a cache passed to two reads
      decoding and making each photograph once, and a photograph changed
      after the read failing the refinement with `RuntimeError` naming it;
      in [test_cluster_patches.py](../../tests/patch/test_cluster_patches.py),
      the command on the `seoul_bull_workspace` fixture writing the file the
      `read_image_rgb` arrays write, byte for byte, and with `mipmap_options`
      making and reading every photograph's mipmap; and in the viewer's
      [cluster_patches/tests.rs](../../crates/sfm-explorer/src/cluster_patches/tests.rs),
      its build equal to the command's steps re-derived, now reading the
      photographs through a cache of their own, in core and with a threshold
      of 64 px, over photographs whose patches pin a position so the members
      are measured. Two parts of the plan were not built: "two calls
      without `cache=` share nothing" has nothing a test can observe, and the
      peak memory measured below is in the open question rather than a test.
      The plan as written:
      `ImagePyramidSet.read`, reading its photographs with `get_many`
      through the `cache=` it is given or a cache of its own, counting
      mipmaps made into `progress=`; the cluster refinement's binding
      (`sfmtool._sfmtool.matching.refine_cluster_patches`) taking an
      `ImagePyramidSet` in place of its `images` list, as the
      `PatchCloud` kernels do; and `sfm cluster-patches` reading through it
      instead of decoding every photograph whole with `read_image_rgb` in a
      thread pool (P4). For a `.matches` file, which names its
      photographs and workspace but has no reconstruction, the command
      passes `read` the list of paths
      `[workspace_dir / name for name in image_names]`, as open question
      20 decides. Tests: a list of paths read in order and a
      `CameraViews` refused with `TypeError`; a cache passed
      to two `read` calls makes and decodes once, its `stats` showing the
      second call's reads as hits, and two calls without `cache=` share
      nothing; a set whose photograph changes after the read raising
      `RuntimeError` naming it from a kernel; `sfm cluster-patches` on the
      `seoul_bull_workspace` fixture writing the same file as before,
      byte for byte; and with the small threshold the CLI and the viewer's
      cluster-patches build (row 19) refining the same members to the same
      values. Peak memory measured before and after on a synthetic
      workspace.
    - **16c, `embed-patches` and the `xform` steps.** **Built.**
      `sfm embed-patches` reads its photographs with
      `ImagePyramidSet.read(recon, cache=cache)` through a cache it makes, in
      [_commands/embed_patches.py](../../src/sfmtool/_commands/embed_patches.py),
      and hands the set to `embed_patches`, which takes a set or a list of
      arrays and cuts a list into a set once; and `apply_transforms`
      ([_apply.py](../../src/sfmtool/xform/_apply.py)) makes one cache for
      the run, when its first step that reads photographs runs, and passes
      it to each such step, which reads its photographs with
      `ImagePyramidSet.read(recon, cache=cache)`. `load_workspace_images`
      is deleted. Choices made in building it: a step that reads photographs
      says so with `reads_photographs = True` and takes
      `apply(recon, cache=None)`, as a step that needs an embedded value says
      so with `required_feature_source`, so the `Transform` protocol's
      `apply(recon)` is unchanged for every other step and a step applied on
      its own reads through a cache of its own; and both commands print the
      `Photographs:` line and the recommendation of open question 21 that
      `sfm cluster-patches` prints, from one function,
      `photograph_cache_lines` in
      [_photograph_cache_report.py](../../src/sfmtool/_photograph_cache_report.py),
      since a run of either reads every photograph in each kernel call or
      step and decodes them again as `sfm cluster-patches` does once their
      tiles are more than the budget. In core both write the same file as
      before, byte for byte; the measured cost is in open question 21,
      with why both took longer from current mipmaps than in core: the
      region decode of an original, since decoded in strips the decoder
      runs on the calling thread, and the process heap (open question 22).
      Tests: in
      [test_embed_patches_command.py](../../tests/patch/test_embed_patches_command.py),
      the command on every eighth point of the `seoul_bull_workspace`
      fixture writing the file the `read_image_rgb` arrays write, byte for
      byte but the write record, with each photograph decoded once, and with
      `mipmap_options` making each mipmap once; in
      [test_photograph_cache.py](../../tests/xform/test_photograph_cache.py),
      the same for `sfm xform --to-embedded-patches --add-patch-bitmaps
      --refine-normals --localize-keypoints --refine-keypoints`, its four
      steps that read photographs sharing the one cache; and in the viewer's
      [state/edits/tests.rs](../../crates/sfm-explorer/src/state/edits/tests.rs),
      *Convert to Embedded Patches* and *Refine All Normals* giving the
      values the core calls behind `--to-embedded-patches
      --add-patch-bitmaps --refine-normals` give over photographs read
      through a cache of their own, in core and with a threshold of 64 px.
      The plan as written:
      `sfm embed-patches`
      and the photometric `xform` steps (`--refine-normals`,
      `--refine-keypoints`, `--localize-keypoints` and the heavy columns),
      which read `load_workspace_images` or `read_workspace_image`
      arrays, on `ImagePyramidSet.read(recon)` (P3), one cache per command
      run. Tests: in-core results unchanged byte for byte on the
      `seoul_bull_workspace` fixture, and with the small threshold equal
      to the viewer's conversion to embedded patches and *Refine all
      normals* for the same reconstruction.
    - **16d, the strips.** **Built.** `sfm compare --strips` and
      `sfm inspect --strips` read their photographs through one
      `PhotographCache` the run makes, in
      [strips/_compare.py](../../src/sfmtool/strips/_compare.py) and
      [strips/_inspect.py](../../src/sfmtool/strips/_inspect.py), shared by
      `compare`'s two reconstructions; the engine in
      [strips/_solve.py](../../src/sfmtool/strips/_solve.py) samples each
      view with `WarpMap.remap_bilinear(set, image_index=i)`, which now takes
      an `ImagePyramidSet` and reads level 0 with `render_source_tile` and
      the bilinear sampler
      ([flow/warp.rs](../../crates/sfmtool-py/src/flow/warp.rs)); and both
      commands print the `Photographs:` line and the recommendation through
      `photograph_cache_lines`, after `--strips: `, as 16c's commands do.
      `read_workspace_image` and its module, which only the strips read in
      `src/`, are deleted, as `load_workspace_images` was in 16c, and the
      tests that read arrays with it read `read_image_rgb`. Choices made in
      building it: the strips sample through the binding they called
      already, `remap_bilinear`, given the set, rather than through
      `OrientedPatch.render_view_tile`, which takes a set too but anchors
      on a keypoint, picks its sampler by rule and reads each tile's
      clipped share from the photograph, none of which a strip uses; and a
      strip reads a photograph only when it first renders a view of it,
      with `ImagePyramidSet.read([path], cache=…)`, as it read one
      `read_workspace_image` array at a time before, rather than
      `read(recon)` for every view, since in core a read decodes every
      photograph it names whole and `inspect --strips` on an
      `embedded_patches` reconstruction reads only the photographs its
      listed points are seen in. The normal refinement, and `inspect
      --strips` on a `sift_files` reconstruction, read every photograph
      once with `read(recon, cache=…)`, and the strips then read that set.
      In core every montage is the one before, byte for byte: on copies of
      the `seoul_bull_workspace` fixture, seven montages (`inspect --strips`
      on the `sift_files` file, tight with every view, on its
      `embedded_patches` conversion and with a point at infinity;
      `compare --strips` in the overview, on one axis without refinement,
      and ranked by NCC without context) were the same files before and
      after; and so were `inspect --strips` and `compare --strips` on the
      user's 8 photographs below. Out of core, with a threshold of 64 px on
      the fixture, the tiles read the full-resolution mip level files
      (quality 95), and a montage without refinement differs from in core by
      at most 4 grey levels (mean 1.0 over the pixels that differ); with
      refinement the turned normals, the NCC labels and so the tiles
      differ more, since the refinement reads the stored levels. On the
      user's 8 photographs (16320 x 12240, region-readable originals),
      the `inspect --strips` montage from current mipmaps is the in-core one
      but for its title, which names its file.
      Measured on the first 8 of the user's photographs with the copied
      solve cut to them (an `embedded_patches` file with bitmaps), one run
      each, wall clock and peak working set, `inspect --strips` of points
      0-15 and `compare --strips` of the file against itself (its
      refinement reading every photograph of both columns):

      | Run | `inspect --strips` | `compare --strips` |
      |-----|--------------------|--------------------|
      | before 16d: `read_workspace_image` arrays | 1.6 s, 1.56 GB | 100.4 s, 21.3 GB |
      | 16d, in core | 1.6 s, 1.69 GB (1 whole decode) | 12.5 s, 7.85 GB (8 whole decodes) |
      | 16d, `mipmap_options`, mipmaps current | 0.4 s, 0.25 GB (0 whole decodes) | 3.2 s, 1.66 GB (0 whole decodes) |

      Before 16d each of `compare`'s two columns decoded every photograph
      for its refinement and cut a set from the arrays, so the 8 were
      decoded 16 times and held twice; with one cache they are decoded once.
      Tests: in
      [test_warp_map_remap_bilinear_rust_bindings.py](../../tests/rust_bindings/flow/test_warp_map_remap_bilinear_rust_bindings.py),
      a photograph of a set in core sampled as its array is, one out of core
      sampled from its full-resolution level and not from the original's
      pixels, and `image_index` required with a set, refused with an array
      and past the set; in
      [test_strips_photographs.py](../../tests/test_strips_photographs.py),
      `compare --strips` reading through one cache, each photograph decoded
      once, and its montage the one a set of `read_image_rgb` arrays
      renders, byte for byte, and, with its two reconstructions in two
      workspaces and a cache too small for their photographs, the
      recommendation naming both workspace files; `inspect --strips` decoding every photograph
      of a `sift_files` reconstruction and only the photographs its listed
      points are seen in for an `embedded_patches` one, both montages the
      arrays' byte for byte; and `inspect --strips` with `mipmap_options`
      making every mipmap and decoding none whole.
      The plan as written:
      `compare --strips` and `inspect --strips`
      (`src/sfmtool/strips/`), which read `read_workspace_image` arrays,
      every photograph at once in `_inspect.py` and one at a time in
      `_solve.py`, on `ImagePyramidSet.read` (P3). Tests: the montages
      unchanged in core.
    - **16e, `write_web_export(cache=)` and the `scripts/` harnesses.**
      **Built.** Core's `build_web_export` and `write_web_export` take the
      `PhotographCache` the bitmap render and the thumbnails read through,
      in place of the budget-0 cache `build_web_export` made for itself
      ([web_export.rs](../../crates/sfmtool-core/src/web_export.rs)); the
      binding `write_web_export(…, cache=None)` passes the cache it is given,
      or makes one of the default budget for the call, as
      `ImagePyramidSet.read` does
      ([fileio/web_export.rs](../../crates/sfmtool-py/src/fileio/web_export.rs));
      `export_reconstruction` passes its `cache` on; and `sfm web-export`
      passes `PhotographCache()`, a cache of the default budget, as open
      question 23 decides. The harnesses read their photographs with
      `ImagePyramidSet.read(recon)`: `track_at_pixel/context.py`, whose
      `images` arrays, which only the plane-sweep candidate reads, are
      decoded the first time one asks;
      `add_image_to_tracks/harness.py` and `misses.py`;
      and `reference_view/measure.py`. Choices made in building it:
      `bench_bars/measure.py` stays on the constructor for every set, the
      plain one included, against the plan below: it plants rows into the
      photographs' arrays and scores each track before and after, so all its
      sets must come from the same arrays, where a plain set from `read`
      would read the stored mip levels once the tier is on, and the script
      decodes every photograph whole for the plants anyway; core takes the
      cache as an argument rather than a field of `WebExportOptions`, as
      `render_display_patch_bitmaps` does, so the options stay a plain
      value; and the command's budget, which open question 23 decides. On
      the `sfm web-export` of the user's 8
      photographs of that question, cut to a file with neither bitmaps nor
      thumbnails so that the export reads every photograph, the files
      written before and after are the same bytes, in core and from current
      mipmaps, at budget 0 and at the default budget. On the Seoul bull
      ground truth, read in place or copied
      by the harnesses that prepare a dataset, `bench_bars/measure.py` (6
      tracks), `reference_view/measure.py` (a pool of 6), `misses.py`,
      `add_image_to_tracks/harness.py` (image 0) and
      `track_at_pixel/harness.py` (5 points) wrote the same rows before and
      after but for their timings, and the track file the same points and
      tracks; the user's other datasets were not run.
      Tests: in
      [web_export/tests.rs](../../crates/sfmtool-core/src/web_export/tests.rs),
      an export through a cache it is given counting every mipmap it makes in
      that cache, and a second export through it making none; and in
      [test_web_export.py](../../tests/test_web_export.py), with
      `mipmap_options` on the copied Seoul bull ground truth, the export
      counting its 17 mipmaps in the cache passed to `write_web_export` and
      `ImagePyramidSet.read` through it making none.
      The plan as written:
      `write_web_export` taking the `cache=` its export reads through, and
      the harnesses that build an `ImagePyramidSet` from `read_image_rgb`
      arrays, `scripts/track_at_pixel/context.py`,
      `scripts/add_image_to_tracks/harness.py` and `misses.py` (§ "Every
      reader of photograph pixels"), and `scripts/bench_bars/measure.py`
      and `scripts/reference_view/measure.py`, found since, on
      `ImagePyramidSet.read`. `bench_bars/measure.py` also builds a set
      from arrays it has modified, which stays on the constructor, since
      `read` reads only the files. Tests: a cache passed to
      `write_web_export` and to `ImagePyramidSet.read` makes each mipmap
      once.
17. **The tier on.** **Built**, in 17a to 17d below, so every step of the
    plan is built. `sfm ws init` writes `mipmap_options` with its three
    values, `sfm ws mipmaps --enable` adds it to an existing workspace, the
    viewer's one-line notice for a workspace without it, the line
    recommending the object (open question 21) that `sfm cluster-patches`,
    `sfm embed-patches`, `sfm xform`, `sfm compare --strips` and
    `sfm inspect --strips` print naming `sfm ws mipmaps
    --enable` in place of the object to add by hand, and the workspace
    spec gains the object. Tests: a new workspace has the object with 1536,
    95 and 95; `--enable` writes the object and refuses a file that has one;
    a workspace without the object writes nothing (the checked-in `test-data`
    workspaces are such workspaces).

    The step is four pull requests, 17a to 17d, each a writer of the object
    or a line that names one, so that no part names a command that does not
    exist yet: the new workspace's object first, then `--enable` with
    `sfm ws upgrade`'s test, then the lines that name `--enable`. The
    `.sfm-workspace.json` version stays 2 in every part: version 2 already
    has the optional `mipmap_options` key, which the current reader reads and
    validates
    ([workspace.md § File Format](../workspace/workspace.md#file-format)),
    and a reader that does not know the key, as sfmtool 0.4.1's
    `load_workspace_config` does not, reads every photograph in core, which
    misreads nothing the file holds.

    - **17a, a new workspace has the object.** **Built**
      ([marker.rs](../../crates/sfmtool-core/src/workspace/marker.rs),
      [mipmap_options.rs](../../crates/sfmtool-core/src/workspace/mipmap_options.rs),
      [_workspace.py](../../src/sfmtool/_workspace.py)), as planned here.
      The one writer of a new `sfmtool` workspace,
      `sfmtool_workspace_marker`, writes `"mipmap_options"` after
      `feature_prefix_dir`, so `sfm ws init`,
      `sfm pick-images --new-workspace` and the viewer's demo workspace get
      it; `init_workspace` writes the same object for the `colmap` and
      `opencv` tools, from the binding `new_workspace_mipmap_options()`; and
      `sfm ws init` prints `mipmap_options: photographs with a side over
      1536 px are read from mipmaps, JPEG quality 95 (95 at full
      resolution)`. `sfm ws init --force` writes a fresh file, so it writes
      the object too. `sfm undistort` writes its output workspace's file
      itself and writes no object until 17b. In `sfmtool_core::workspace`:

      ```rust
      /// The `mipmap_options` a new workspace records: 1536, 95 and 95.
      pub const NEW_WORKSPACE_MIPMAP_OPTIONS: MipmapOptions;
      /// The object as `.sfm-workspace.json` holds it, which
      /// `read_mipmap_options` reads back to `options`.
      pub fn mipmap_options_json(options: &MipmapOptions) -> serde_json::Value;
      ```

      and in Python `sfmtool.workspace.new_workspace_mipmap_options() ->
      dict`, the object with its keys in the written order. Tests: the
      default and custom markers end with the object, byte for byte as
      `json.dump(..., indent=2)` writes it; `read_mipmap_options` reads a
      new workspace's object back as `NEW_WORKSPACE_MIPMAP_OPTIONS`; a
      workspace `sfm ws init` makes for each of the three tools has the
      object with 1536, 95 and 95, and the command prints it.
    - **17b, `sfm ws mipmaps --enable`.** **Built**
      ([mipmap_options.rs](../../crates/sfmtool-core/src/workspace/mipmap_options.rs),
      [ws.py](../../src/sfmtool/_commands/ws.py),
      [_undistort_images.py](../../src/sfmtool/_undistort_images.py),
      [ws-mipmaps-command.md § Turning the tier on](../cli/workspace/ws-mipmaps-command.md#turning-the-tier-on)),
      as planned here, with these choices the plan left open. The object is
      added as the file's last key by inserting its text before the closing
      brace, so the rest of the file keeps its bytes, and a file
      `json.dump(..., indent=2)` wrote reads as the one it would write with
      the object last; it is written through a temporary file and a rename.
      The values are checked by `parse_mipmap_options`, the reader's check
      made public, which the binding calls on the Python integers before
      converting them, so a quality of 300 gets the reader's message, not an
      `OverflowError`. `WorkspaceConfigError` gains `MipmapOptionsPresent`
      and `Version`: a file whose `version` is not 2 is refused, since the
      object is a key of a version 2 file. `--enable` takes exactly one
      `PATH`, and `--enable` with `--check` or `--prune`, or one of the
      three values without `--enable`, is a usage error. `sfm undistort`
      reads the source's object through a new binding,
      `sfmtool.workspace.read_mipmap_options`, before any image is
      undistorted. The plan as written:
      `sfm ws mipmaps --enable PATH`
      adds the object to the workspace at or above `PATH`, with
      `--out-of-core-long-side`, `--jpeg-quality` and
      `--full-resolution-jpeg-quality` defaulting to
      `NEW_WORKSPACE_MIPMAP_OPTIONS`' values, and refuses a file that has
      the object, naming it; it makes no mipmap, and says that
      `sfm ws mipmaps PATH` makes them now and that every reader makes those
      it needs. It keeps every other key of the file and its order. In Rust,
      `enable_mipmap_options(root: &Path, options: &MipmapOptions) ->
      Result<(), WorkspaceConfigError>`, with a variant for a file that has
      the object, bound as `sfmtool.workspace.enable_mipmap_options`. Tests:
      `--enable` writes the object with the defaults and with each option,
      keeping the file's other keys; it refuses a file that has one and
      leaves it as it is; `--enable` with `--check` or `--prune` is a usage
      error; `sfm ws upgrade` of a workspace without the object writes none,
      and of one with it keeps it.

      17b also has `sfm undistort` carry the object over (open question
      25): the `.sfm-workspace.json` it writes in its output directory gets
      the source workspace's `mipmap_options`, unchanged and after
      `feature_prefix_dir`, when the source's file has one, and none when it
      has none. Tests: undistort
      from a workspace with an object whose values differ from
      `NEW_WORKSPACE_MIPMAP_OPTIONS` writes that object, and from one
      without writes none; `undistort-command.md` § "Workspace config" shows
      the key.
    - **17c, the recommendation names `--enable`.** **Built**
      ([_photograph_cache_report.py](../../src/sfmtool/_photograph_cache_report.py),
      [cluster-patches-command.md § Process](../cli/image-feature/cluster-patches-command.md#process)),
      with these choices the plan left open. The line names each workspace
      the in-core photographs are in, found from each photograph by
      `find_workspace_for_path`, the photograph cache's own rule, so
      `photograph_cache_lines` no longer takes the callers' workspace
      directories; the workspaces are named in sorted order, since `xform`
      and `compare` pass their paths as a set. A photograph is out of core
      when a side is over the threshold, so the threshold that puts a
      workspace's photographs out of core is one less than the shortest of
      their longer sides. For a workspace without the object the line names
      `sfm ws mipmaps --enable <workspace>`, with
      `--out-of-core-long-side <threshold>` when that threshold is under
      the default 1536, which would leave such a photograph in core: the
      Seoul bull's 270 x 480 photographs get
      `--out-of-core-long-side 479`. For a workspace whose object has a
      threshold that keeps the photographs in core, which `--enable`
      refuses since the file has the object, it says to set
      `"out_of_core_long_side"` in the workspace's `.sfm-workspace.json` to
      that threshold or less, and names the value the file holds. This is
      the case of every workspace `sfm ws init` makes since 17a whose
      photographs are 1536 px or smaller, such as the tests' Seoul bull
      workspaces. A workspace is shown as the `sfm ws` commands show a
      path, relative to the current directory when it is under it, by
      `shown_path` in [_cli_utils.py](../../src/sfmtool/_cli_utils.py),
      moved there from `sfm ws`. The plan as written: the line
      `photograph_cache_lines` prints for `sfm cluster-patches`,
      `sfm embed-patches`, `sfm xform`, `sfm compare --strips` and
      `sfm inspect --strips` names `sfm ws mipmaps --enable <workspace>` for
      each workspace without the object, in place of the object to add by
      hand. Tests: the line names the command and the workspace.
    - **17d, the viewer's notice.** **Built**
      ([photograph_tier_notice.rs](../../crates/sfm-explorer/src/state/photograph_tier_notice.rs),
      [action-log.md § What gets logged](../gui/action-log.md#what-gets-logged),
      [photograph-cache.md § In the viewer](../core/camera/photograph-cache.md#in-the-viewer)),
      with these choices the plan left open. The viewer says it as one
      Action Log row of its own, kind `Session`, as it reports the system
      font search, so the 3D view's status line shows it until the next
      entry; nothing has to be dismissed. The cache notes the photograph:
      `request_layout` and `read_layout` keep the first one they find with a
      side over `DEFAULT_OUT_OF_CORE_LONG_SIDE` in a workspace whose file
      has no object, and `PhotographCache::large_photograph_without_mipmap_options`
      answers it, reading nothing and taking no lock, so the frame asks
      every time, before the MCP drain, and `AppState` remembers that the
      session has said it. A photograph in no workspace is not one, nor is
      one of a workspace whose object has a threshold that keeps it in core.
      The notice comes when a panel first reads such a photograph's layout,
      as when an image is selected, not at open. The plan as written: the
      first time in a session the
      photograph cache's layout read finds a photograph with a side over
      1536 px in a workspace without the object, the viewer says once, with
      the text the draft's § "Mip level files" gives it, *This workspace
      decodes large photographs whole. Run `sfm ws mipmaps --enable` to read
      them a tile at a time.* Tests: the notice is given once for two such
      photographs and not for a workspace with the object.
      The docs site changes in pull requests to
      `sfmtool/sfmtool.github.io`: beside 17a, the getting-started
      tutorial's `sfm ws init` output, which ends with the new
      `mipmap_options` line; beside 17b, the pages that describe
      `sfm ws mipmaps`.

## Measurements

The scratch program behind the figures, kept out of the tree: a binary crate
depending on `jpeg-encoder = "=0.7.1"` (feature `simd`) and
`jpeg-decoder = "=0.3.2"` (no default features), built `--release`. It decodes
the photograph, makes the levels with a copy of `downsample_2x`, encodes level 1
at quality 90 and 95, 4:4:4 and 4:2:0, with restart intervals of 0, 8, 16 and
64 MCUs, decodes each back (time, mean absolute error against the exact level),
encodes every level at both qualities, indexes the restart markers of the
original and of level 1, and decodes 512 px regions as above, comparing each
with the whole decode pixel by pixel. For `-mip0` it encodes the whole
decode at qualities 90, 95, 97, 98 and 100 (4:4:4, interval 16 MCUs), as it is
and cropped by one pixel on the top and left, and compares the decode with the
source. Times are wall clock of one run each, repeated two to four times; the
ranges in this draft are over those runs.

Step 6's figures come from a test of sfmtool-core's own code instead, run once
in the test profile (optimized, with debug information) and kept out of the
tree: it reads one of the user's photographs, builds level 1 with
`downsample_2x` and encodes it as a mip level file at qualities 90 and 95, and
times five runs of each tile and region decode, comparing each with `decode`
or `read_rgb` pixel by pixel.

## Open questions

The user has decided every question below; the section records the
decisions.

1. **The names of the mip level files.** *Decided by the user:* the whole
   original file name followed by the suffix, as a `.sift` is named
   (`IMG_0001.JPG.sift`), which cannot collide, then `-mip<l>-q<quality>.jpg`
   with the level number and the JPEG quality (`IMG_0001.JPG-mip1-q95.jpg`
   for half size, `-mip0-q95.jpg` for the full-resolution copy), following
   the standard mipmap convention in which levels are numbered and level 0 is
   the base image (§ "Terms"), and making a change of quality unambiguous.
2. **The threshold's value and the other constants.** *Decided by the
   user:* `sfm ws init` writes `out_of_core_long_side` 1536, changed from
   4096 by open question 26. The tile size and the restart interval stay
   constants.
3. **Qualities.** *Decided by the user:* `sfm ws init` writes 95 for both
   `jpeg_quality` and `full_resolution_jpeg_quality`, with no rule that
   depends on the original's format; a workspace changes them by editing its
   object.
4. **The Python SIFT path and the mipmap build.** *Decided by the user:* they
   stay separate. The mipmap build decodes the file itself rather than being
   handed the array SIFT decoded, which is simpler and cannot be handed the
   wrong pixels, at the cost of a second decode per out-of-core photograph
   (0.8 to 1.7 s at 200 MP).
5. **The command's name.** *Decided by the user:* `sfm ws mipmaps`, since the
   files belong to the workspace.
6. **The build budget and cancellation.** *Decided by the user:* (a) for
   both, as designed. Builds are bounded by reservations made at decision
   time on each cache's non-blocking count, each kernel caller still granted
   one when the budget is used up and a panel's decode refused and asked
   again on a later frame, with builds in the viewer made only by background
   tasks, the panels' needs going to the *Build mipmaps* quiet task (the
   budget plus one mipmap per running task, two in the viewer); not a
   blocking semaphore, which would make a rayon worker wait for another
   caller's permit, and not a count per `get_many`, which would let two tasks
   each take the budget. A cancelled maker abandons its build and a waiter takes it over,
   rather than every maker finishing.
7. **Image Detail's zoom limit.** *Decided by the user:* the limit is
   `max(32, 8 / fit_scale)` times the fit, so a level-0 pixel can always be
   enlarged to 8 screen pixels (§ "Image Detail").
8. **Mip level files written by another build.** *Decided by the user:* accepted. A
   file is the record, so a newer `jpeg-encoder` never rewrites a current
   mip level file, though a read-only copy of the workspace read through the
   in-memory equivalent by another build can differ from the files by the
   encoders' differences; `encoder` stays in the record for diagnosis.
9. **The decoder.** *Decided by the user:* the mipmap reader keeps
   `jpeg-decoder` (whose `ColorTransform::None` the tile decode needs, and
   whose upsampling and conversion the region decode of an original relies
   on) until zune-jpeg passes the same tile-equality tests. If `read_rgb`
   moves to zune-jpeg, as
   [reading-and-writing-images.md](../core/camera/reading-and-writing-images.md)
   plans, and the region decode no longer equals it for some layout, the
   region test sends that layout to `-mip0`, as a stated change of
   results.
10. **Budgets.** *Decided by the user:* ship the defaults (the cache's one
    budget of a quarter of memory, 1 to 16 GiB, and 256 GPU tile textures), report
    the tiles decoded and reused in each kernel's phase note and the cache's
    `stats` in the Background Tasks panel's tooltip, and tune them on real
    captures.
11. **SIFT and the kernels read different level 0s** for an original that is
    not region-readable. *Decided by the user:* accepted. SIFT reads the
    original and the kernels `-mip0` (0.07 to 0.09 grey levels apart for
    the user's JPEGs, 0.19 to 0.32 for the non-JPEG proxy); SIFT runs before
    the files exist and its output is stored.
12. **Restart intervals wider than two tiles.** *Decided by the user:* (a),
    as designed. The region test refuses intervals wider than 1024 px, so
    such originals get a `-mip0` (33 to 43 MB and one encode, once) and
    every tile after it reads in about 3 ms, rather than every full-resolution
    read decoding a strip of the whole width.
13. **`sfm ws upgrade` and the object.** *Decided by the user:* (a), as
    designed. `sfm ws upgrade` leaves an absent `mipmap_options` absent, and
    `sfm ws mipmaps --enable` adds the object; adding it changes which pixels
    the kernels read for every large photograph, a change the owner makes by
    a command named for it. The viewer's notice names the command
    ([mipmaps.md § "The workspace setting"](../core/camera/mipmaps.md#the-workspace-setting)).
14. **What makes the mipmaps panels queue before step 15.** Found while
    splitting step 14. Step 14e has the panels queue a
    photograph whose mipmap is not current on *Build mipmaps*' shared list
    and draw *Building mipmap…* until its layout is `Known`, and step 15
    builds the task that makes them. Between the two, in a workspace whose
    owner wrote `mipmap_options` by hand, nothing takes photographs off the
    list, so a panel showing such a photograph draws the spinner until a
    kernel's read phase happens to make its mipmap. The options:
    - (a) 14e also builds the task as far as the list needs it: a quiet
      source on the fast queue whose job takes one chunk from the front of
      the list, calls `get_many` on it and ends, queued again while the list
      is not empty; step 15 then adds the order at open and the rest of its
      test list.
    - (b) the panels' queueing moves from 14e to step 15, beside the task,
      and 14e has the panels draw *Building mipmap…* for a photograph that
      is `Building` without queueing it.
    - (c) as planned: 14e queues and nothing drains the list until step 15,
      accepted since the tier is off.

    *Decided by the user:* (a). 14e builds the job and the shared list, the
    job taking one chunk from the front of the list, calling `get_many` on
    it and ending, queued again while the list is not empty, so no part
    leaves a panel waiting on work nothing will do; step 15 adds the rest
    of the task (§ "Implementation steps", 14e and 15). Since the viewer
    already cancels every quiet task running in the queue an asked-for task
    is queued in, 14e's job gives way from the first, so 14e also puts a
    cancelled chunk's unfinished photographs back at the front of the list
    and gives the source its `QuietRank` place and its answer to
    `waits_for_its_node`: without them a photograph lost in a cancelled
    chunk would never be queued again, and its panel would draw the spinner
    for good, which (a) was chosen to rule out. This follows from (a) and
    needs no further decision.
15. **A panel's read of an evicted level held in memory.** Found while
    building 14c: when a mipmap's files cannot be written, its levels are
    held in memory as file entries, evicted under the budget and made again
    when a kernel reads them. A panel that asks for tiles of such a level
    after it was evicted (on a zoom or a pan; a panel keeps the tiles it is
    answered, so not on every frame) could be answered in three ways:
    - (a) from a whole decode of the original in core that no slot keeps,
      reserved with `try_reserve_build`, as for a photograph without a
      current mipmap, so a panel never makes a mipmap;
    - (b) by the panel making the levels again in memory without waiting
      (reserved with `try_reserve_build`, answering `Busy` when refused or
      when the slot is being made, and writing no files), so later requests
      cost nothing until the levels are evicted again;
    - (c) by keeping the panel's in-core tiles in the slot, which would mix
      pixels one JPEG loss apart into what kernels read.

    *Decided by the user:* (a), as built in 14c. It happens only in a
    workspace whose mipmap files cannot be written, with the cache under
    its budget, and it keeps the rule that a panel never makes a mipmap.
    Step 14e, which gives panels the `Building` state, looks at it again.
    Looked at again in 14e1: (a) still fits. Answering such a read
    `Building` instead would have *Build mipmaps* make the levels again
    in memory for the panel, only for them to be evicted again, and (b)
    and (c) are still ruled out by the rule that a panel never makes a
    mipmap and by what kernels read. 14e1 applies (a) also under a budget
    of 0, where no slot keeps the levels at all, so that a panel's every
    request would otherwise have the build made again (14e1's as-built
    note).
16. **Gestures on the coarsest level drawn while a mipmap is made.** Found
    while building 14e2a: Image Detail draws the selected photograph's
    coarsest level under the spinner while *Build mipmaps* makes its
    mipmap. The options:
    - (a) it takes no click, drag or menu, as G13 has a picture kept while
      its replacement loads take none, hover and the view's own wheel and
      keys still working;
    - (b) it takes the gestures the finished photograph does, under G1 to
      G12, only the painted spinner and its label sitting over it.

    *Decided by the user:* (b). The level is the selected photograph at low
    resolution, drawn at its real level-0 size with the overlays lined up,
    so a click, a drag or a menu names the pixel and the feature the user
    sees; G13's reason, that the drawn picture is not the selected one,
    does not apply. The first form of 14e2a built (a) and named the
    stand-in in G13; that clause is gone and the panel's spec cites G1 to
    G12 for it.
17. **The binding `sfm sift` and `sfm ws mipmaps` build through in step
    15.** Found while splitting step 15. Step 15 has `sfm sift` and
    `sfm ws mipmaps` make mipmaps, and step 16 moves "`sfm ws mipmaps` and
    the Python SIFT extractors on `PhotographCache.build_mipmaps`", the one
    way to build mipmaps from Python with no free function (§ "Python
    bindings"). So step 15 names no binding for them to call. The options:
    - (a) 15d brings forward the `sfmtool.photographs` submodule with the
      `PhotographCache` class, its constructor and `build_mipmaps` (and
      `stats`, which a test of shared builds reads), wired as § "Python
      bindings" says; step 16 adds `layout`, `read_level`, `read_tile` and
      `read_region`, `ImagePyramidSet.read` and the readers moved onto it;
    - (b) step 15 adds a free binding, such as
      `sfmtool.workspace.build_mipmaps(paths)`, and step 16 replaces it
      with the class's method, deleting the free function;
    - (c) `sfm sift`'s mipmaps and the whole of `sfm ws mipmaps` move to
      step 16, beside the class, and step 15 keeps the Rust `.sift` build,
      the viewer's list at open and the thumbnail.

    *Decided by the user:* (a). It keeps the rule that the class's method
    is the one way to build from Python, builds no binding only to delete
    it, and keeps the makers in the step that is about makers; the class is
    small without its reads, and its wiring (the submodule, the
    lazy-loading and registration tests) is work step 16 would otherwise
    do. Step 15d brings the submodule with the constructor,
    `build_mipmaps` and `stats`, and step 16 adds the reads.
18. **`sfm web-export`'s thumbnail of an out-of-core photograph whose
    mipmap is not current.** Found while splitting step 15.
    `display_thumbnail_row` fills a missing thumbnail column for both the
    viewer's open and `sfm web-export`, "so the two show the same picture".
    § "Readers on the GUI thread" decides the viewer's case: the
    placeholder row at open, filled from the coarsest mip level file once
    the photograph is `Known`. A web export writes its thumbnails once, so
    it has no later fill. The options:
    - (a) the export makes the mipmap first, through the photograph cache
      it reads with (`get_many`, as it makes the mipmaps its patch
      bitmaps need), and reads the coarsest file, so its rows are the
      viewer's;
    - (b) the export decodes the original whole and resizes it, as today,
      so its row differs slightly from the viewer's;
    - (c) the export writes the placeholder row.

    *Decided by the user:* (a), as § "Who makes the mipmaps, and when" has
    every reader of the cache make the mipmaps it needs, and the export
    already makes them for its patch bitmaps. Step 15c builds it.
19. **A closed node's photographs on *Build mipmaps*' list.** Found while
    building 15b: the open puts a whole reconstruction's out-of-core
    photographs without a current mipmap on the list, which can be minutes
    of builds, and the plan does not say what happens to them when the node
    is closed first. The options:
    - (a) drop on close: closing a node, or Close All, takes every
      photograph that no node left in the scene has off the list, so no
      later run makes it, and `sfm ws mipmaps` (15e) prepares a workspace
      whose reconstruction is not open;
    - (b) keep building: the list keeps a closed node's photographs and the
      task goes on making them, as the 14e1 list did.

    *Decided by the user:* (a), as built in 15b
    (`AppState::forget_closed_mipmaps`). A chunk a run has already taken
    is left to finish, but a cancelled run does not put a closed node's
    photographs back unless an open or a panel queued them again first.
20. **What `ImagePyramidSet.read` takes for `sfm cluster-patches`.** Found
    while splitting step 16. § "Python bindings" gave
    `ImagePyramidSet.read(views_or_recon, workspace_dir=None, cache=None,
    progress=None)`, and step 16 moves `sfm cluster-patches` onto it. But
    that command reads a `.matches` file, which names its photographs
    (`image_names`, under its workspace) and their sizes (`image_dims`) and
    has no reconstruction and no cameras; and a `CameraViews` names no
    photographs, so `read` has no path to read for one either. The
    options:
    - (a) `read`'s first argument is an `SfmrReconstruction`, whose
      `image_names` are read under `workspace_dir` (its own workspace when
      omitted), or a list of photograph paths, read in that order with no
      check of their sizes at the read (a kernel checks the set against its
      scene, as it does today); a `CameraViews` is refused, since it names
      no photographs. `sfm cluster-patches` passes
      `[workspace_dir / name for name in image_names]`.
    - (b) `read` keeps a scene as its first argument, for the size check,
      and takes the paths as a second, `paths=None`, required with a
      `CameraViews`; `sfm cluster-patches` builds a `CameraViews` from the
      file's `image_dims`, with poses it does not use.
    - (c) no `ImagePyramidSet` for it: the cluster refinement's binding
      takes `image_paths=` and `cache=` and reads through `get_many`
      itself, as `write_web_export` does, against § "Python bindings"'
      rule that the kernels take arrays or a set, not paths.

    *Decided by the user:* (a). It reads the same photographs the
    viewer's cluster-patches build reads (row 19, `get_many` over the
    paths), and it invents no scene for a command that has none. The
    signature is `ImagePyramidSet.read(recon_or_paths, workspace_dir=None,
    cache=None, progress=None)`, as § "Python bindings" gives it, and step
    16b builds it.
21. **`sfm cluster-patches` with in-core photographs past the cache's
    budget.** Found while building 16b. Until step 17
    turns the tier on, a workspace has no `mipmap_options` and every
    photograph is in core, so the cache keeps whole decodes' tiles, and a
    tile it has evicted is read by decoding the whole photograph again. The
    refinement takes its clusters in the file's order, in parallel, and each
    reads several photographs, so once the photographs' tiles are more than
    the budget, they are decoded again many times. Measured on the first 8
    of the user's 36 photographs (16320 x 12240), copied to a scratch
    workspace, with the 24,805 clusters (63,492 members) of the user's
    clusters file that lie among them; each figure is one run, wall clock,
    with the process's peak working set:

    | Run | Cache budget | Whole decodes | Time | Peak working set |
    |-----|--------------|---------------|------|------------------|
    | before 16b: `read_image_rgb` arrays | none | 8 | 6.4 s | 14.4 GB |
    | 16b, in core | 15.9 GiB (default) | 8 | 5.3 s | 8.8 GB |
    | 16b, in core | 3.3 GiB | 65 | 36 s (61 s in a second run) | 8.9 GB |
    | 16b, in core | 0.9 GiB | 408 (446) | 188 s (279 s) | 8.9 GB |
    | 16b, `mipmap_options` (4096, 95, 95), making the mipmaps | 15.9 GiB | 0, 8 mipmaps made | 7.0 s | 7.6 GB |
    | 16b, `mipmap_options`, mipmaps current | 15.9 GiB | 0 | 4.4 s | 2.3 GB |
    | 16b, `mipmap_options`, mipmaps current | 0.9 GiB | 0 | 7.0 s | 1.4 GB |

    In core every 16b run wrote the file of the run before 16b, byte for
    byte (with the timestamp fixed); with the tier the runs agree with each
    other, and 882 of the 63,492 members take another status than in core,
    since they read the stored levels.

    The tiles of one photograph are about 0.85 GB, so the user's 36 are
    about 30 GB against the default budget of 16 GiB on the 64 GB machine,
    about the 3.3 GiB row's ratio. Before 16b the command held all 36 as
    arrays and tiles at once, about 1.8 GB each, 65 GB in all. The peak
    working set does not fall below about 9 GB however small the budget,
    because the decodes in progress are bounded by the build budget (16 GiB
    on that machine), not by the cache's budget. The options:
    - (a) take the clusters in an order that reads each photograph's tiles
      together: the kernel, or the command, groups the clusters by a
      member image and refines a group at a time. Grouping by the lowest
      member image and refining each group in its own kernel call, tried in
      a scratch script, decoded about as often (62 against 67 at 3.3 GiB,
      421 against 446 at 0.9 GiB) and was no faster, since a group's
      clusters still reach most photographs and rayon runs a group's
      clusters across every thread; an order that helps would have to
      bound the photographs a stretch of clusters reads;
    - (b) a budget for the command that holds every photograph when memory
      allows, such as the photographs' tile bytes up to half of physical
      memory, which keeps the time and holds about 0.85 GB per 200 MP
      photograph, still half of what the arrays held, but needs about 30 GB
      for the user's 36;
    - (c) leave it, and recommend the tier for photographs of this size
      (`mipmap_options` in the workspace file now; step 17 makes `sfm ws
      init` write it): with it the same run reads tiles of the mip level
      files and decodes no photograph whole, at any budget.

    *Decided by the user:* (c). 16b ships as built, relying on the tier.
    When the command's cache decodes photographs whole more times than the
    in-core photographs it read, the command prints a line after its
    `Photographs:` line recommending a `mipmap_options` object in the
    workspace's `.sfm-workspace.json`, naming the file and giving the object
    to add by hand; step 17 changes it to name `sfm ws mipmaps --enable`.
    Step 16c moves `sfm embed-patches` and the `xform` steps onto the same
    reads, and they print the same lines (§ "Implementation steps", 16c).
    Measured as above on the same 8 photographs with the copied solve cut
    to them (4,755 points, 9,411 observations): `sfm xform --refine-normals
    --localize-keypoints --refine-keypoints`, and `embed_patches` with its
    defaults, given the set `sfm embed-patches` reads (the command takes a
    `sift_files` reconstruction, which the copied solve is not):

    | Run | Cache budget | Whole decodes | Time | Peak working set |
    |-----|--------------|---------------|------|------------------|
    | before 16c: `read_image_rgb` arrays, each step decoding every photograph | none | 24 | 51.5 s (178 s in a second run) | 14.6 GB (14.4 GB) |
    | 16c, in core | 15.9 GiB (default) | 8 | 19.2 s (62 s) | 8.4 GB (8.0 GB) |
    | 16c, in core | 3.2 GiB | 133 | 94 s | 11.3 GB |
    | 16c, in core | 0.8 GiB | not finished; stopped after about 45 min | | |
    | 16c, `mipmap_options` (4096, 95, 95), mipmaps current | 15.9 GiB | 0 | 81 s | 3.2 GB |
    | 16c, `mipmap_options`, mipmaps current | 0.8 GiB | 0 | 150 s | 3.4 GB |

    | Run | Cache budget | Whole decodes | Time | Peak working set |
    |-----|--------------|---------------|------|------------------|
    | before 16c: `read_image_rgb` arrays, decoded in a thread pool | none | 8 | 61 s | 14.4 GB |
    | 16c, in core | 15.9 GiB (default) | 8 | 54 s | 7.9 GB |
    | 16c, in core | 3.2 GiB | 184 | 354 s | 11.6 GB |
    | 16c, `mipmap_options` (4096, 95, 95), mipmaps current | 15.9 GiB | 0 | 169 s | 3.9 GB |
    | 16c, `mipmap_options`, mipmaps current | 0.8 GiB | 0 | 437 s | 2.2 GB |

    In core every 16c run wrote the file of the run before 16c, byte for
    byte but the write record. Before 16c each `xform` step decoded every
    photograph again, one after another, so the three steps decoded them
    three times; with one cache for the run they are decoded once at the
    default budget. The machine was shared with other work while these
    ran, and the same run took up to three times as long from one run to
    the next, so the times compare only roughly. Unlike `sfm cluster-patches` in 16b, both
    commands took longer reading their photographs out of core, from
    current mipmaps, than in core at the default budget (81 s against 19 s
    for the `xform` steps, 169 s against 54 s for `embed_patches`), while
    holding less than half the memory. What cost the time was found before
    step 17, below.

    **Where the time went.** Measured on the same workspace copies, by
    counting and timing the cache's reads in a build that was not committed
    (each figure the thread time summed over the 32 rayon threads), and then
    by sampling the threads' states:

    - *A region of a region-readable original was decoded on threads that
      could not run.* `jpeg-decoder` decodes a frame wider than 128 px on one
      new OS thread per colour component and waits for them, and a level-0
      tile's frame was about 600 px wide. While the kernel's own threads keep
      every core busy those threads wait their turn: the 1,817 level-0 tiles
      of an `xform` run took 368 s, 203 ms each, and the 2,429 of an
      `embed_patches` run 539 s, 222 ms each, against 3 ms for the same tile
      on an idle machine and 66 ms on one thread beside 32 busy ones. The
      mip level files' tiles were not affected, since their frames are
      128 px wide (559 level-1 tiles took 11 s, 20 ms each). Each original was
      also read and indexed 23 to 31 times for 8 photographs, by threads
      that asked for it at once, 4 to 25 s in all; and the coarse levels'
      decodes, 4 to 5 of them, under 1.2 s. Reads answered from kept tiles
      took 5.5 s and 28 s in all, and under 0.2 s once the fix below was
      made. Fixed: a region is decoded in strips of frames no wider
      than 128 px, which the decoder decodes on the calling thread
      ([core/camera/mipmaps.md § "A region of a region-readable original"](../core/camera/mipmaps.md#a-region-of-a-region-readable-original)),
      about 13 ms a tile on one idle thread for about twice the pixels; the
      level-0 tiles of the same runs then took 25 s (17 ms each) and 52 s
      (23 ms each). Every 512 px tile of one of the photographs equals the
      whole decode, and every output below is the same file before and
      after, byte for byte but the write record.
    - *The process heap.* The fix left `embed_patches` from mipmaps at about
      52 s against 37 s in core in the same session, though its reads now
      took 66 s of thread time, about 2 s of the run, and the in-core run's
      `ImagePyramidSet.read` 2.6 s. Run again over the same cache, every read
      a kept tile, it still took 46 to 65 s against 26 to 48 s in core, with
      160 to 310 s of kernel-mode time against 17 to 26 s, 24.8 million
      context switches against 2.3 million, and a third of its threads'
      samples waiting on a lock (`WrAlertByThreadId`) against a tenth. On one
      thread it took the time the in-core run took (17.5 s on every sixteenth
      point), and at quality 100 the time it took at 95, so the kernel's work
      was the same. In one process, after one run from the mipmaps whose
      set and cache were then dropped, the in-core run took 62 to 93 s
      against 34 to 42 s in a fresh one. With `mimalloc` as the extension's
      global allocator, in an uncommitted build, the run from the mipmaps
      took 28.9 s with its reads and 28.3 s without against 28.2 s in core,
      with 6 to 10 s of kernel-mode time, and a run from the mipmaps no
      longer slowed the in-core runs after it (27.9 to 28.2 s). The thousands
      of tile decodes on the kernel's own threads, each allocating and
      freeing blocks of tens to hundreds of KB, leave the Windows process
      heap, which Rust's default allocator uses, slow for every thread that
      allocates after them. Changing the allocator is open question 22.

    Measured as above, wall clock of one run each, by the build before the
    fix (`a19c12b2`), with it (`aa076827`), and with it and `mimalloc` (an
    uncommitted experiment); several values are runs repeated in the same
    session, and the machine's other work made the same run vary by up to
    two times, so the lowest is the better guide:

    | Run | Cache budget | Before the fix | With the fix | With the fix and `mimalloc` |
    |-----|--------------|----------------|--------------|-----------------------------|
    | `xform`, in core | 15.9 GiB | 21.7, 35.6 s | 23.0 to 56.9 s (6 runs) | 17.8 s |
    | `xform`, mipmaps current | 15.9 GiB | 28.3, 48.1 s | 26.1 to 37.1 s (6 runs) | 17.1 s |
    | `xform`, mipmaps current | 0.8 GiB | 60.4 s | 63.4 s | 36.9 s |
    | `embed_patches`, in core | 15.9 GiB | 39.0, 41.2 s | 37.1 to 123.7 s (6 runs) | 31.7 s |
    | `embed_patches`, mipmaps current | 15.9 GiB | 73.5, 76.4 s | 47.1 to 79.0 s (6 runs) | 31.6 s |
    | `embed_patches`, mipmaps current | 0.8 GiB | 126.2 s | 106.0 s | 61.9 s |
    | `sfm cluster-patches`, in core | 15.9 GiB | 6.1, 7.1 s | 7.1 to 10.6 s (4 runs) | 5.5 s |
    | `sfm cluster-patches`, mipmaps current | 15.9 GiB | 3.3, 4.7 s | 4.7 to 26.7 s (4 runs) | 3.9 s |
    | `sfm cluster-patches`, mipmaps current | 0.8 GiB | 6.0 s | 8.4 s | 5.6 s |

    The peak working set was unchanged by the fix (3.1 GB for `xform` and
    3.8 GB for `embed_patches` from the mipmaps, 8 to 9 GB in core); with
    `mimalloc` it was 3.5 and 4.3 GB from the mipmaps at the default budget,
    4.7 and 5.0 GB at 0.8 GiB (3.1 and 3.9 GB without it), and 10.4 to
    10.7 GB in core. `sfm cluster-patches` reads mostly the stored levels,
    whose tiles were never slow, which is why it was faster from the
    mipmaps before the fix. At 0.8 GiB the runs from the mipmaps still take
    about twice the default budget's time: the level-0 file entries are the
    originals' whole bytes, about 60 MB each, so the eight photographs'
    entries and the tiles their kernels use are more than the budget, and an
    `xform` run reads the originals 271 times and decodes 30,523 level-0
    tiles, against 27 and 1,725 at the default budget.
22. **The global allocator.** Found while measuring open question 21. With
    the reads of open question 21 fixed, a kernel reading photographs from
    their mipmaps still took up to 1.5 times as long as in core, because the
    tile decodes on its threads leave the Windows process heap slow for
    every allocation after them (the measurements are in open question 21).
    The options:
    - (a) `mimalloc` as the global allocator of the extension
      (`sfmtool-py`) and of the viewer's own binary: a dependency that
      builds C code with the `cc` crate on every platform, MIT-licensed.
      Measured: from the mipmaps as fast as in core (31.6 s against 31.7 s
      for `embed_patches`, 17.1 s against 17.8 s for `xform`), and in core
      about 20% faster than with the system allocator, but the peak working
      set 10 to 50% higher (5.0 GB against 3.9 GB for `embed_patches` at
      0.8 GiB), which the cache's budget does not count.
    - (b) fewer allocations in the tile decode: reuse a thread's buffers for
      the small JPEG and the decoded strip, and decode a tile straight into
      the tile's pixels. `jpeg-decoder` still allocates its own buffers for
      every frame, so this lessens the churn without removing it; not
      measured.
    - (c) leave it: from current mipmaps the commands hold a third to a half
      of the in-core memory and take up to 1.5 times as long on Windows.

    *Decided by the user:* (a). `mimalloc` becomes the global allocator of
    `sfmtool-py` and of the viewer's own binary, after the same runs are
    measured on Linux and macOS, since their allocators differ from
    Windows' heap. It is a change of its own, made before step 17.

    *Built, measured on Windows and Linux; macOS not measured:* `mimalloc`
    0.1.52 (mimalloc 3.3.2) is the global allocator of `sfmtool-py`
    ([lib.rs](../../crates/sfmtool-py/src/lib.rs)) and of the
    `sfm-explorer` binary
    ([main.rs](../../crates/sfm-explorer/src/main.rs)), unconditionally,
    with no Cargo feature to turn it off; the library crates set none
    ([python-bindings.md § "The global allocator"](../python-bindings.md#the-global-allocator)).
    The extension enables `mimalloc`'s `local_dynamic_tls` feature: built
    without it, the Linux extension failed to import, with "cannot allocate
    memory in static TLS block", since Python loads it with `dlopen`.
    Measured as in open question 21 on the same 8 photographs, by a scratch
    script reading wall clock, user and kernel-mode CPU time (`os.times`)
    and the peak working set (Windows) or `ru_maxrss` (Linux). On Windows,
    two runs each of a release build at this change's base with the system
    allocator and the same build with `mimalloc`, the two builds
    alternating run by run; each cell is wall clock, kernel-mode time and
    peak working set, the two runs separated by "/":

    | Run | Cache budget | System allocator | `mimalloc` |
    |-----|--------------|------------------|------------|
    | `xform`, in core | 15.9 GiB | 22.9 / 23.5 s; 18 / 19 s; 8.3 / 8.5 GB | 18.8 / 18.4 s; 8 / 7 s; 10.4 / 10.9 GB |
    | `xform`, mipmaps current | 15.9 GiB | 31.5 / 33.0 s; 85 / 86 s; 3.1 / 3.1 GB | 17.4 / 17.5 s; 9 / 8 s; 3.6 / 3.6 GB |
    | `xform`, mipmaps current | 0.8 GiB | 63.6 / 51.1 s; 72 / 85 s; 2.2 / 2.9 GB | 37.6 / 37.1 s; 46 / 47 s; 4.4 / 3.4 GB |
    | `embed_patches`, in core | 15.9 GiB | 37.7 / 38.5 s; 23 / 29 s; 8.3 / 8.4 GB | 31.4 / 31.6 s; 10 / 11 s; 10.5 / 10.3 GB |
    | `embed_patches`, mipmaps current | 15.9 GiB | 59.4 / 46.2 s; 212 / 100 s; 3.8 / 3.8 GB | 30.5 / 30.5 s; 11 / 11 s; 4.3 / 4.3 GB |
    | `embed_patches`, mipmaps current | 0.8 GiB | 80.6 / 106.7 s; 132 / 211 s; 3.7 / 4.2 GB | 62.7 / 63.4 s; 74 / 72 s; 3.9 / 5.5 GB |
    | `sfm cluster-patches`, in core | 15.9 GiB | 6.7 / 6.6 s; 7 / 8 s; 8.7 / 8.8 GB | 5.5 / 5.6 s; 5 / 6 s; 10.5 / 10.3 GB |
    | `sfm cluster-patches`, mipmaps current | 15.9 GiB | 4.4 / 4.5 s; 5 / 6 s; 2.3 / 2.3 GB | 4.0 / 4.1 s; 5 / 5 s; 2.7 / 2.8 GB |
    | `sfm cluster-patches`, mipmaps current | 0.8 GiB | 6.8 / 6.8 s; 11 / 11 s; 1.3 / 1.4 GB | 5.9 / 5.8 s; 6 / 6 s; 1.8 / 1.8 GB |

    On Linux, Ubuntu 24.04 under WSL 2 on the same machine, with 31 GB and
    32 threads, so the default cache budget is 7.8 GiB, one run each, with
    glibc's `malloc`, with `mimalloc`, and with `mimalloc` and
    `MIMALLOC_PURGE_DELAY=0`; each cell is wall clock, kernel-mode time and
    peak resident set:

    | Run | Cache budget | glibc | `mimalloc` | `mimalloc`, purge delay 0 |
    |-----|--------------|-------|------------|---------------------------|
    | `xform`, in core | 7.8 GiB | 32.7 s; 31 s; 8.7 GB | 21.6 s; 7 s; 10.6 GB | 26.6 s; 118 s; 9.6 GB |
    | `xform`, mipmaps current | 7.8 GiB | 20.0 s; 12 s; 3.2 GB | 19.8 s; 9 s; 3.8 GB | 25.4 s; 126 s; 3.3 GB |
    | `xform`, mipmaps current | 0.8 GiB | 49.1 s; 89 s; 4.8 GB | 42.2 s; 59 s; 4.7 GB | 50.8 s; 330 s; 3.3 GB |
    | `embed_patches`, in core | 7.8 GiB | 56.0 s; 14 s; 9.2 GB | 36.9 s; 13 s; 10.9 GB | 44.3 s; 197 s; 9.2 GB |
    | `embed_patches`, mipmaps current | 7.8 GiB | 82.2 s; 40 s; 3.9 GB | 35.6 s; 11 s; 4.6 GB | 43.7 s; 199 s; 4.1 GB |
    | `embed_patches`, mipmaps current | 0.8 GiB | 70.7 s; 127 s; 5.2 GB | 69.2 s; 89 s; 4.5 GB | 89.1 s; 556 s; 2.8 GB |
    | `sfm cluster-patches`, in core | 7.8 GiB | 17.3 s; 53 s; 9.2 GB | 16.7 s; 48 s; 10.2 GB | 16.4 s; 57 s; 9.0 GB |
    | `sfm cluster-patches`, mipmaps current | 7.8 GiB | 15.0 s; 47 s; 2.3 GB | 14.2 s; 44 s; 2.8 GB | 15.3 s; 67 s; 2.4 GB |
    | `sfm cluster-patches`, mipmaps current | 0.8 GiB | 17.1 s; 49 s; 1.9 GB | 16.8 s; 48 s; 2.0 GB | 18.1 s; 90 s; 1.5 GB |

    `MIMALLOC_PURGE_DELAY=0` on Windows, one run each, not alternating:
    `xform` 26.9 s in core, 24.5 s and 64.8 s from the mipmaps at the two
    budgets, with 67, 83 and 217 s of kernel-mode time and 8.6, 3.2 and
    3.1 GB; `embed_patches` 50.7 s, 42.8 s and 105.0 s, with 151, 153 and
    383 s and 8.6, 4.0 and 2.7 GB. With a purge delay of 100 ms,
    `embed_patches` from the mipmaps at 0.8 GiB took 97.5 s with 3.9 GB.
    Purging at once brings the peak back to the system allocator's but
    costs most of the gain where the cache evicts, so the default delay of
    1 s stays. Every output was the same file with either allocator on each
    platform, byte for byte but the write record. On both platforms the
    runs from the mipmaps at the default budget take the time of the
    in-core runs or less, and `xform` and `embed_patches` in core are 15
    to 35% faster than with the system allocator; at 0.8 GiB the runs still take about twice
    the default budget's time, for the reason open question 21 gives.
    `sfm cluster-patches` under WSL 2 spends about 45 s of kernel-mode time
    with either allocator, which these runs did not look into. macOS was
    not measured, since no Mac was available. *Decided by the user:* ship
    it with Windows and Linux measured, and macOS covered by the CI's
    macOS build and test jobs, which build the extension and the viewer
    with `mimalloc` and run the full suites; macOS can be benchmarked
    later.
23. **The cache budget of `sfm web-export`.** Found while building 16e. The
    command reads through a cache of budget 0, as
    [view-region-pyramid.md](view-region-pyramid.md) open question 11
    decided: an answer holds the tiles of the decode that made it, so an
    in-core photograph is decoded once. An out-of-core photograph's answer
    holds no tile, so under budget 0 every tile the bitmap render reads is
    decoded from its mip level file again. Measured on the first 8 of the
    user's photographs (16320 x 12240) with the copied solve cut to them
    and saved without its patch bitmaps and thumbnails, so the export
    renders the bitmaps and makes the thumbnails from the photographs; one
    run each, wall clock and peak working set, on the shared machine, whose
    in-core times varied from 6 to 14 s between runs of the same code:

    | Run | Cache budget | Time | Peak working set | Misses in `stats` |
    |-----|--------------|------|------------------|-------------------|
    | before 16e, in core | 0 | 6.0 s | 10.5 GB | |
    | 16e, in core | 0 | 11.6 s, 13.7 s | 10.0 GB, 10.0 GB | 8, each a whole decode |
    | 16e, in core | 15.9 GiB (default) | 8.2 s | 14.9 GB | 8, each a whole decode |
    | before 16e, mipmaps current | 0 | 74.0 s | 4.4 GB | |
    | 16e, mipmaps current | 0 | 41.2 s, 64.1 s | 4.7 GB, 4.6 GB | 9,931 |
    | 16e, mipmaps current | 15.9 GiB (default) | 1.2 s | 4.5 GB | 2,016 |

    Every run wrote the same files. In core the default budget's cache
    still held 6.4 GB of tiles at the end, and the thumbnails, made after
    the render, decode each in-core photograph again outside the cache, so
    both are held at once. The options:
    - (a) the command passes a cache of the default budget: from current
      mipmaps the export takes about a second, and in core its peak rises
      by about the cache's tiles of the photographs, here 4.8 GB. Once step
      17 turns the tier on, a photograph in core is at most 1536 px on its
      long side, so the rise is large only in a workspace without it;
    - (b) keep budget 0, as built: the in-core peak stays, and an export
      from mipmaps takes about a minute for these 8;
    - (c) extend open question 11's rule: under budget 0 an out-of-core
      answer holds the tiles and file entries it has read until it is
      dropped, as an in-core answer holds its decode. Budget 0 then reads
      each tile once in either tier; it is a change to the cache in core,
      a step of its own.

    *Decided by the user:* (a). `sfm web-export` passes `PhotographCache()`,
    a cache of the default budget, which supersedes the budget 0 that
    [view-region-pyramid.md](view-region-pyramid.md) open question 11 chose
    for the export (that decision's rule for answers under budget 0 stands
    for every other caller). From current mipmaps the export of these 8
    takes about 1.2 s rather than 41 to 64 s, and in core its peak rises by
    about 4.8 GB, from 10.0 to 14.9 GB; the tier is how the draft bounds
    memory for photographs of this size. Re-measured after the change, one
    run each, through the command: in core 10.0 s and 14.98 GB, from
    current mipmaps 1.7 s and 4.09 GB, each writing the files of the run
    before 16e, byte for byte.
24. **The workspace file's version.** Found while splitting step 17. Writing
    `mipmap_options` into every new workspace could be marked by a new
    version of `.sfm-workspace.json`, as a new optional entry of one of the
    archive formats is ([format-versions.md](../formats/format-versions.md)).
    *Decided by the user:* the file stays at
    version 2. `mipmap_options` is already an optional key of version 2,
    which the current reader reads and validates, and sfmtool 0.4.1's
    `load_workspace_config` ignores keys it does not know, so it reads a new
    workspace's file and decodes every photograph whole, which misreads
    nothing the file holds
    ([workspace.md § File Format](../workspace/workspace.md#file-format)).
25. **`sfm undistort`'s output workspace.** Found while building 17a.
    `sfm undistort` writes the `.sfm-workspace.json` of its output directory
    itself, not through `init_workspace`, so 17a gives it no object. The
    options: (a) copy the source workspace's object, (b) write
    `NEW_WORKSPACE_MIPMAP_OPTIONS`, as a new workspace does, (c) write none.
    *Decided by the user:* (a). The output workspace gets the source
    workspace's `mipmap_options` when the source's file has one, and none
    when it has none, so undistorting keeps the choice the source's owner
    made; it is part of step 17b.
26. **The default threshold.** Found while reviewing 17a. Open question 2
    first set the `out_of_core_long_side` a new workspace records to 4096:
    an in-core photograph's whole decode, made again each time its tiles are
    evicted, starts to cost a few tenths of a second there, and it was the
    side at which a new workspace stops doubling images for SIFT, so SIFT
    never doubled an out-of-core photograph. *Decided by the user:* 1536,
    since the benefits for high image counts are important too: a job that
    reads many photographs of a few megapixels, such as those of a
    24-megapixel camera (6000 x 4000) or a phone (4032 x 3024), reads them a
    tile at a time, so its memory stays bounded where in core it holds whole
    decodes. `DEFAULT_OUT_OF_CORE_LONG_SIDE` is 1536, and
    `NEW_WORKSPACE_MIPMAP_OPTIONS`, the line `sfm ws init` prints and the
    object the recommendation line of open question 21 gives all take it
    from there; the viewer's notice of 17d is given for a photograph over
    1536 px, and 17b's `--enable` defaults to 1536.
    `double_image_below_long_side` stays 4096, so a new workspace's
    photograph whose longer side is 1537 to 4095 px is out of core and
    doubled for SIFT; nothing relies on the two agreeing (§ "Which
    photographs are out of core"). A workspace already made with 4096
    keeps it, as every workspace keeps its own value. Scripts that make a
    new workspace of
    `dino_dog_toy`'s 2040 x 1536 photographs, `scripts/solve_crossval.py
    --datasets dino` and the setup `scripts/benchmark_flow_matching.py`
    describes, now have them out of core, so the extraction also makes their
    mipmaps; the `.sift` files are unchanged. The checked-in `test-data`
    workspaces have no object, and the `init_dataset_*.sh` scripts make a
    workspace only where the file is missing, which for `dino_dog_toy` it
    is not.
