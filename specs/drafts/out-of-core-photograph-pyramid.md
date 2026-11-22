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
(§ "Open questions", 1 to 13) and every one of the companion draft's seven.

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
| **out-of-core photograph** / **in-core photograph** | a photograph of a workspace whose file has `mipmap_options`, with a width or height over that object's `out_of_core_long_side` (4096 as `sfm ws init` writes it), is out of core (§ "Which photographs are out of core"; one with a side of 65536 px or more is refused everywhere): its reduced levels come from mipmaps, its level 0 from the original or a `-mip0`; any other photograph is in core: it has no mip level files, and its tiles are cut from a whole decode of the original, made when a tile of it is needed and not kept. Every photograph, in core or out, is read a region at a time through photograph tiles | `large image`, `tiled image`, `big photograph` | the tier decides where the levels live, on disk or only in memory, which is what *core* means here; *large* says nothing about what is done differently, and every photograph is read in tiles |
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
limit (step 16). The test data, 3840 x 2160 video frames and 4032 x 3024
phone photographs are in core under the default threshold; a 24-megapixel
camera (6000 x 4000) and the 200-megapixel phone photographs are out of core.

Why 4096 rather than 8192, the texture limit: once every photograph is drawn in
tiles (below), the texture limit no longer separates anything, and the tier is
about the time of a read. An in-core photograph whose tiles were evicted is
decoded whole again, about 0.05 s at 3840 x 2160 but 0.2 to 0.4 s for a
6000 x 4000 photograph (estimated by pixel count), while an out-of-core one
decodes only the tiles it needs, about 3 ms each, at the cost of one extra
encode of its levels, once.
4096 is also where `sfm ws init` stops SIFT image doubling
(`double_image_below_long_side`: an image is doubled only when both sides are
below 4096), so at the default SIFT never doubles an out-of-core photograph.
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
  invalid `mipmap_options` is not read as out of core until the file is
  fixed; its readers report the file's error (Image Detail says *Workspace
  file error:* and the message). `load_workspace_config` calls
  `read_mipmap_options` through a binding, raising `RuntimeError` (step 16).
- **Writing the object** (steps 15 and 17): `sfm ws init` writes it with 4096,
  95 and 95; `sfm ws mipmaps --enable` adds it to an existing workspace (or
  with `--out-of-core-long-side`, `--jpeg-quality` and
  `--full-resolution-jpeg-quality`) and refuses a file that has it;
  `sfm ws upgrade` leaves it as it finds it; and the viewer says once per
  session, the first time it reads a photograph over 4096 px in a workspace
  without the object: *This workspace decodes large photographs whole. Run
  `sfm ws mipmaps --enable` to read them a tile at a time.*
- **Orphans** (step 15): `sfm ws mipmaps --prune` deletes the mip level files
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
  test and the index of its 195,075 intervals, took 10 to 12 ms.
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
other entry. A tile or region held by a running render stays alive until the
render drops it, as a pyramid does today. A budget of 0 keeps nothing: each
request is decoded and handed to its asker, whose texture or render keeps it.

**States, and what a caller on the GUI thread may do.** `peek_tile` and
`layout` answer without reading or waiting:

```rust
pub enum PeekedLayout {
    Known(Arc<PyramidLayout>), // header read, and for out of core files current: tiles may be asked for
    Building { coarsest: Option<Arc<PhotographTile>> }, // mipmap missing, stale or being made; the coarsest once published
    Unreadable(String),         // the original or the workspace file cannot be read; the message says which
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
  (open question 4): after each `.sift` is written, the extractor calls
  `PhotographCache.build_mipmaps` (§ "Python bindings") for the photograph,
  on one cache the extraction run holds, so its worker threads share one
  build count, and that call decodes the photograph itself. So on the command
  line the mipmaps cost a second decode (0.8 to 1.7 s at 200 MP) besides the
  reduction and the encodes (about 0.8 s, plus 1.4 to 2.1 s for a
  `-mip0`, estimated from the parts above). Only the Rust `build_sift_file`
  in [sift/files.rs](../../crates/sfmtool-core/src/features/sift/files.rs),
  behind the viewer's *Build SIFT files* and `pick-images`, makes them from
  the decode it already holds, passing it as `build_mipmaps`'s `decoded`
  argument, since there the decode is in Rust and handing it over is a
  borrow, not a binding's array.
- **`sfm ws mipmaps PATHS…`** makes them explicitly for the photographs under
  the paths (skipping in-core ones and current mipmaps), with `--prune` to delete
  orphans and `--check` to report stale mipmaps without writing. It is the way to
  prepare a workspace whose features came from somewhere else.
- **Any reader of the cache** makes the mipmaps it needs, in any process, as
  § "Building a mipmap in the slot" says. On the command line and in Python,
  `ImagePyramidSet.read(recon, progress=…)`, `embed-patches`,
  `sfm cluster-patches` and `sfm web-export` read through a cache whose
  `get_many` makes every missing mipmap in its phase, counting mipmaps through the
  caller's `Progress` (a `ProgressCounter` from Python), before the work
  starts; `sfm ws mipmaps` calls `get_many` over every photograph under its
  paths in a phase *build mipmaps*.
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
  the Image Browser strip shows, then the rest in image order. A panel that
  needs a photograph whose mipmap is not current **adds it to the front of
  the list**: the next run takes it in its chunk, the running chunk being
  left to finish, and when no run is running or waiting one is queued, under
  the quiet-task rule that a source keeps at most one waiting entry. While it
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
| The 3D background | the same layout request; nothing drawn | queues the photograph; draws the coarsest level if published | asks for its level's tiles with `try_decode_tiles` on the rayon pool, then reads the level with `get_level` once they are present (under a budget of 0, builds it from the tiles that call returns), and draws the coarsest until it lands |
| Display thumbnails | built in the open job, off the GUI thread: a photograph whose coarsest mip level file is not current gets the grey placeholder row | the row stays the placeholder | a placeholder row whose photograph has become `Known` is filled, on the rayon pool, from the coarsest mip level file |

A build publishes each reduced level to the slot's `Building` state as it is
encoded, coarsest first, so a panel shows a photograph's coarsest level about a
second after the build starts, before its finer levels and its `-mip0` are
written.

**A budget of 0.** `display_photograph` with `SFM_EXPLORER_PHOTOGRAPH_CACHE_MB=0`
calls `get` on the GUI thread today for the panels that still call it, since
nothing it decodes would be kept for the next frame; Image Detail has not since
step 13c2. No panel decodes on the GUI thread in this design, at any
budget: with a budget of 0, a tile decoded on the rayon pool is handed to the
panel that asked for it, whose texture keeps it, the cache keeps nothing, and
an out-of-core photograph is still queued for *Build mipmaps*.

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
for with `display_photograph`, as
[gui/camera-views.md](../gui/camera-views.md) § "Photograph background"
describes; the level rule, the texture coordinates, the alpha and the
remembered failure are as below. Until a photograph lands, the previous
image's photograph stays on its own mesh (nothing is drawn before any has
landed), since there is no coarsest level to draw before the tiled cache.
Step 13c3 swaps the source for the level's tiles and `get_level`, inside the
closure the app passes to `upload_bg_image` and its `Decoded` answer:

- **Level.** The coarsest level whose longer side is still at least the
  viewport's longer side in physical pixels, but no finer than the first level
  whose longer side fits `max_texture_dimension_2d`. At 3840 px of viewport
  that is level 2 (4080 x 3060) of the 200-megapixel photograph; at 1920 px,
  level 3.
- **Source.** The level assembled from the photograph's tiles, once the
  photograph is `Known`, for every photograph. On the rayon pool the
  background first asks for the level's tiles with `try_decode_tiles`,
  asking again on a later frame when the answer is `Busy`, then reads the
  level with `get_level` once they are present; under a cache budget of 0 it
  builds the level from the tiles that call returns. So no panel task waits
  on a slot or on the build count, and that rule has no exception (decided by
  the user, the companion's open question 9). The coarsest level is drawn
  until it lands. A photograph that cannot be read is remembered as
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
differ slightly from a thumbnail of level 0; it is for display only.
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
| 11–13 | Display patch bitmaps at open, conversion to embedded patches, `sfm web-export` (`display_bitmaps.rs`, `state/edits.rs` `convert_to_embedded_patches_job`, `web_export.rs`) | `get_many`, full depth, every photograph held at once; web-export with budget 0 | `get_many` answers tiled photographs; regions; web-export makes missing mipmaps in the workspace | the stored levels |
| 14–18 | The bench's steps and searches, live evaluation, *Find Nearby Tracks*, *Create Track Here*, *Add image to tracks*, *Refine all normals*, *Refit all points* (`bench.rs`, `state/edits.rs` `fetch`, `bench/nearby_tracks.rs`, `bench/track_at_pixel.rs`, `add_image_to_tracks.rs`, `state/edits/refine_all_normals.rs`, `refit_all_points.rs`) | `get_many`, levels per sample or per grid, `clipped_share` on level 0; the normal kernels' renders, the grown plane's own sampler and the cluster refinement's `LevelTile` copies | regions, `clipped_share` as in row 10; the grown plane's reads through `read_source_samples` and `LevelTile` through `read_source_rect` (the companion's § "Readers that sample through their own code") | the stored levels |
| 16a | *Find Nearby Tracks*' grey reads (`read_grey`, `bench/nearby/grey.rs`) | each read blurs the level-0 rectangle it samples, plus the blur's 4 px, through `read_source_rect` over the whole pyramid `from_pyramid` wraps (built, step 12); before step 12, level 0 of each view, blurred grey `f32`, whole: 800 MB per 200-megapixel view | the same rectangles, from the cache's tiles | level 0 from `-mip0` for an original that is not region-readable |
| 19 | The viewer's cluster-patches build (`cluster_patches.rs` `read_photographs`) | `get_many` through the cache (built, 11b), full depth, every photograph held at once | `get_many` answers tiled photographs; regions | the stored levels |
| 20 | Display thumbnails when none is stored (`display_thumbnails.rs` → `display_thumbnail_row`) | whole decode, resized to 128 | in core unchanged; out of core from the coarsest mip level file | display only |
| 21 | `.sift` building (`features/sift/files.rs` `build_sift_file`) | whole decode | whole decode of the original, then the mipmap from it | no |
| 22 | Image readings at save (`reconstruction/image_readings.rs` `gather_image_readings`) | whole decode of every photograph with no `.sift` row, with the number decoded at once bounded by memory as SIFT's is (built in step 4) | unchanged | no |
| P1 | `sfmtool.fileio.read_image_rgb` / `read_image_rgba` (`sfmtool-py/src/fileio/image.rs`) | whole decode | unchanged: they read files, not pyramids | no |
| P2 | `ImagePyramidSet(recon, images)` and the kernels' `images` lists (`sfmtool-py/src/patches/views.rs`) | exact pyramids of the arrays passed | tiled photographs cut from the arrays (`TiledPhotograph::from_image`), the same pixels; a new `ImagePyramidSet.read(recon)` reads by path through the core cache | no for arrays; through `read`, as rows 11–18 |
| P3 | `embed-patches`, the strip montages, the `xform` steps that read photographs (`src/sfmtool/_embed_patches.py`, `src/sfmtool/strips/`, `src/sfmtool/xform/_images.py`) | `read_workspace_image` arrays into the kernels | `ImagePyramidSet.read(recon)` | the stored levels, so they match the viewer |
| P4 | `sfm cluster-patches` (`src/sfmtool/_cluster_patches.py`) | `read_image_rgb` arrays, full depth | through the same reader as #19 | the stored levels |
| P5 | The view-tile binding (`sfmtool-py/src/patches/view_tile.rs`) | an exact pyramid of the array passed | a tiled photograph cut from the array | no |
| P6 | Optical flow's pyramids (`sfmtool-py/src/flow/warp.rs`) | a grey pyramid of its own from an array | unchanged; flow reads level 0 from its array | no |
| P7 | Spherical tile source stacks (`sfmtool-py/src/spherical/tile_source_stack.rs` → `spherical/per_tile_source_stack.rs`) | `remap_bilinear` of the arrays passed | unchanged: they read arrays, not pyramids | no |

The measurement harnesses under `scripts/` that build an `ImagePyramidSet` from
`read_image_rgb` arrays (`track_at_pixel/context.py`,
`add_image_to_tracks/harness.py` and `misses.py`) are P2 callers: they read
the arrays' exact levels until they move to `ImagePyramidSet.read`, which step
16 does. A P2 caller that passes arrays holds them whole, as today; that is the
caller's choice of input, and `read` is the bounded way.

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
pub const DEFAULT_OUT_OF_CORE_LONG_SIDE: u32 = 4096; // what `sfm ws init` writes
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
/// reading the files back.
pub fn build_mipmaps(
    photograph: &Path,
    decoded: Option<&ImageU8>,
    options: &MipmapOptions,
    tile_size: u32,
    progress: &Progress<'_>,
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
    /// Tiles cut from the image's pyramid of `full_levels` depth, built by
    /// `ImageU8Pyramid::build` as the bindings build it today, so the same
    /// pixels; owned by the source, outside any cache: the Python bindings
    /// given arrays, and tests.
    pub fn from_image(image: ImageU8) -> Self;
    /// A whole pyramid the cache handed out, read in place: a tile or region
    /// read copies only the pixels it asks for from the pyramid's levels, and
    /// the source keeps no tile, so a reader that moves holds no more than
    /// the pyramid it held before. Made in step 7, read by moved readers from
    /// step 9, and deleted in step 13e, when the cache hands out tiled
    /// photographs itself.
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
pub enum TileError { Changed { path: PathBuf }, Unreadable { path: PathBuf, why: String },
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
(a `&ProjectedImage`, and so a `&TiledPhotograph`) and the region pool the
kernel's caller made for this kernel call (`RegionPool::for_kernel_call()`,
whatever way the caller fetched its sources; a kernel that calls another
passes its own pool down), and returns `Result<_, PhotographReadError>`:
it works out the regions the read needs and copies them from the pool or
reads through tiles, either of which can fail for a photograph the cache
reads, and never fails for one built `from_image`. No kernel computes a
region or touches a photograph tile.

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
| 11 | the public `render_patch_bitmap_column` / `render_display_patch_bitmaps` in [display_bitmaps.rs](../../crates/sfmtool-core/src/patch/display_bitmaps.rs), which return `PhotographReadError` instead of `Cancelled`, and `WebExportError`, which gains `Photograph` (built, 11a); the viewer's bulk jobs (the conversion to embedded patches, *Refine all normals*, *Refit all points*, the cluster-patches build) and `sfm web-export` take tiled photographs from `get_many` once step 13e makes it answer them, wrapping its whole pyramids with `from_pyramid` until then, and the cluster-patches build fetches through the cache from 11b (built) |
| 12 | `read_patch_along_ray` and `depth_layers`, which take the caller's `&RegionPool` in place of the `&GreyImages` cache, and `far_field_sweep` and `find_nearby_tracks`, which took the pool in 10a and drop the cache; `PatchReadError` and `DepthLayerError` gain `Photograph(PhotographReadError)` in place of `GreyMismatch`, and `GreyImages`, with its `get` and `GreyImageError::NoSuchImage`, is gone (built). The plan also listed `guided_matches`, `nearby_cluster_tracks`, `constellation_seeds`, `nearby_points` and `distance_range`, which read the reconstruction, its index files and the cameras but no photograph, so their signatures do not change |
| 13 | `PhotographCache::get`, which takes the caller's `&Progress`, and `GetError`, which gains `Cancelled` (built, 13a); `PhotographCache::new`, which takes `PhotographCacheOptions` (built, 13b); the layout and tile requests, `TileError` and `TryDecode`, the four that can decode a whole photograph taking the caller's `&Progress` (built, 13c1); `display_photograph`, which gives way to `display_layout` and `display_tiles` (built for Image Detail, 13c2; 13c3 and 13d for the other panels); `get`, `get_many`, `peek` and `peek_state`, which answer `Arc<TiledPhotograph>`, `insert`, which takes an `ImageU8`, `PhotographReadError`, which gains `Changed`, `NotFound`, `TooLarge` and `Unreadable` with the conversions from `TileError` and `GetError`, and `from_pyramid`, which is deleted (13e) |
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
  out-of-core photographs among `paths` (directories expanded as
  `sfm ws mipmaps` expands them, skipping `.sfm-mipmaps`) through the cache's
  `get_many`, so a mipmap already made in this cache, or by another cache or
  process, is
  read rather than made again, and the build budget and its reservations
  apply. It returns a dict, as the bindings' other reports are dicts
  (`write_web_export`'s, `pick_images`' through `serde_to_py`, the bench's),
  with the keys `made`, `current` and `in_core` (lists of the paths in each
  case, as given) and `failed` (a list of `(path, message)` pairs). **It is the one way to
  build mipmaps from Python**; there is no free function for it. `sfm ws mipmaps` makes a
  `PhotographCache()` and calls it; the Python SIFT extractors call it after
  each `.sift` is written, on one cache the extraction run holds (§ "Who
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
reconstruction's photographs by path**: `ImagePyramidSet.read(views_or_recon,
workspace_dir=None, cache=None, progress=None)` and
`sfmtool.fileio.write_web_export(…, cache=None)`. The kernels take arrays or
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
only, once every reader takes them, in seven pull requests: the slot and the
build budget first, then the cache's tile requests and the panels, then the
tiles. Steps 14 to 16 build the out-of-core tier
without turning it on anywhere but in tests: until step 17, neither
`sfm ws init` nor any other command writes `mipmap_options`, so no workspace
has the object unless its owner writes it by hand, and every photograph stays
in core. Step 17 turns the tier on.

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
   pyramids as today and reading them through `from_pyramid`; they peek
   tiles from step 13d. **Built**, before step 8: each panel owns a region
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

    Only the cache can hand out a tiled photograph, and it hands out whole
    pyramids until step 13e, so what these callers fetch changes there: until
    then each wraps the whole pyramids `get_many` answers with
    `from_pyramid`, as the bench's `ViewSources` does, and step 13e's
    `get_many` answers `Arc<TiledPhotograph>` to all of them at once. The
    companion's step-5 tests that count decodes, and its test of a budget
    smaller than one photograph's level 0, need the tiled cache, so they come
    with step 13e. Before it, the step is three pull requests, 11a to 11c:

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
      13e.

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
    from; `get_level`; `display_layout` and `display_tiles` in place of
    `display_photograph`, so Image Detail, the 3D background and every other
    reader read tiles from the cache; Image Detail asking for the next image's tiles a frame ahead while
    animation plays, and `get_image_detail_view`'s `level` and
    `tiles_pending`; `from_pyramid` deleted.

    The step is seven pull requests, 13a to 13e, with 13c in three. Once the
    cache keeps tiles in place of pyramids, no reader can be handed a whole
    pyramid, so every reader that still takes one (Image Detail, the 3D
    background, Track View, the Track Table, the bulk callers that wrap with
    `from_pyramid`) has to move before that, and the slot and the build
    budget, which every decode goes through, come first. So the split runs
    from the slot up: 13a makes the slot cancellable, 13b bounds the decodes
    in progress by the build budget, 13c1 adds the cache's layout and tile
    requests, 13c2 moves Image Detail to them, and 13c3 and 13d move the
    other panels to them, while the cache
    still holds whole pyramids and cuts each tile
    it hands out from one, and 13e replaces the pyramids with tiles, which
    every reader then reads. Until 13e the cache decodes as it does today, so
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
    - **13c3, the 3D background's level.** In place of
      `display_photograph`, the 3D background first asks for its level's
      tiles with `try_decode_tiles` on the rayon pool, asking again on a
      later frame when the answer is `Busy`, then reads the level with
      `get_level` once they are present; under a cache budget of 0 it builds
      the level from the tiles that call returns. So no panel task waits on a
      slot or the count, and the rule has no exception (decided by the user,
      the companion's open question 9). It draws the same pixels as before.
      Tests: the level is asked for once while it loads, a `Busy` answer is
      asked again on a later frame, and nothing is decoded on the GUI
      thread, with a budget of 0 too.
    - **13d, Track View and the Track Table.** Track View's patch tiles,
      cluster-stage tiles and hover views, and the crops of both panels and
      their hover views, act on the layout state and peek tiles, asking for
      the missing ones with `try_decode_tiles`, and render only once every
      tile they need is there, so no panel reads a whole pyramid any more and
      the viewer's `peek_full_res_pyramid` goes. A render reads from a source
      that holds exactly the tiles the panel peeked, pinned for the render,
      so a tile evicted between the peek and the render cannot be read and
      the GUI thread never decodes (the companion's open question 7).
      `peek` and `peek_state` stay for the callers that only ask whether a
      photograph is decoded (the image menu's *Add image to tracks* check),
      and answer tiled photographs from 13e. Tests: the panels' tests of
      their pictures, which compare them with the whole pyramid's, pass unchanged; a cell whose
      tiles are not all there draws its loading state and asks for them
      once; no panel decodes on the GUI thread.
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
      `ValueError` of a refusal, since a read can fail from here on. Tests:
      an in-core photograph's tiles equal the same pixels of today's pyramid;
      a request for one tile decodes the photograph once and leaves its
      other tiles as unused, evicted first; a tile evicted and asked for
      again decodes the photograph again, counted in `stats`; a kernel's
      `get_many` over 20 in-core photographs decodes them all into tiles in
      its read phase, so its renders decode nothing when the tiles fit the
      budget; a `get_many` over 20 photographs holds no pyramid (the cache's
      stats show only tiles, within a small budget); a bulk kernel whose
      photographs' tiles do not fit a small budget still completes, with its
      decodes counted, and so does one with a budget smaller than one
      photograph's level 0; and the companion's step-5 test of the point
      order: on a rayon pool of one thread, with a budget smaller than the
      reconstruction's tiles but holding the tiles of every photograph one
      reference image's points are seen in, each in-core photograph is
      decoded at most once in the read phase and once more for each
      reference image whose points it sees, and the same points in shuffled
      order decode more.
14. **The out-of-core tier, off.** Photographs of a workspace with
    `mipmap_options` read through their mip level files: tiles from the
    files, mipmaps made in the slot when missing or stale (the misses of a
    `get_many` in groups sized by the build budget), `TileError::Changed`,
    and panels queueing photographs whose mipmap is not current for the
    *Build mipmaps* quiet task (the task with its shared list, as far as step
    15 needs it). Tests: with the small threshold, a `get` on a photograph
    with no mipmap makes it and returns its pixels; two threads asking for
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
15. **Makers**: `sfm sift` and `build_sift_file`, the viewer's *Build
    mipmaps* quiet task, `sfm ws mipmaps` with `--prune`
    and `--check` (its `--enable` waits for step 17, since it writes the
    object), and the thumbnail from the coarsest mip level file.
    Tests: `sfm sift` on a small-threshold synthetic workspace writes the
    files; the quiet task runs on the fast queue, makes the selected image's
    mipmap in its first run, ends after each chunk and is queued again while
    its list is not empty, takes a photograph a panel adds to the front of
    its list in its next run without cancelling the running one, lets a
    waiting quiet task of higher `QuietRank` (a camera model proposal's job)
    start between its chunks while it has a long list, pushes no version,
    runs on a locked node, and gives way to an asked-for task, leaving no slot
    marked as being made;
    `--prune` deletes an orphan and
    keeps a current mipmap, and deletes every mip level file in a workspace without the
    object; `--check` writes nothing.
16. **Python and the readers that must agree with the viewer**: the
    `sfmtool.photographs` submodule with the `PhotographCache` class (its
    layout, level, tile and region reads, `build_mipmaps`, `stats`) and core's
    `read_layout` and `read_region` behind it; `ImagePyramidSet.read` and
    `write_web_export` with `cache=`; `sfm ws mipmaps` and the Python SIFT
    extractors on `PhotographCache.build_mipmaps`; and `embed-patches`, the
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
    new submodule passes `tests/test_module_layout.py`, the registration
    test and the lazy-loading test.
17. **The tier on.** `sfm ws init` writes `mipmap_options` with its three
    values, `sfm ws mipmaps --enable` adds it to an existing workspace, the
    viewer's one-line notice for a workspace without it, and the workspace
    spec gains the object. Tests: a new workspace has the object with 4096,
    95 and 95; `--enable` writes the object and refuses a file that has one;
    a workspace without the object writes nothing (the checked-in `test-data`
    workspaces are such workspaces).

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
   user:* `sfm ws init` writes `out_of_core_long_side` 4096. The tile size
   and the restart interval stay constants.
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
