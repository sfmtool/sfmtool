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
original whole"; a departure listed in § "Problems found with the
decisions"); the reduced levels of
a large photograph are JPEG files in `.sfm-cache/` at the workspace root, one
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
(§ "Open questions", 1 to 13); the five open questions of the companion draft
remain open.

Amends [core/camera/photograph-cache.md](../core/camera/photograph-cache.md)
(the cache gains photograph tiles and an on-disk tier, and its pyramids a depth
set by image size),
[core/camera/reading-and-writing-images.md](../core/camera/reading-and-writing-images.md)
(a second JPEG writer and reader, for mip level files),
[gui/multi-panel-image-browser.md](../gui/multi-panel-image-browser.md)
§ "Image Detail" (the panel draws photograph tiles),
[gui/mcp-server.md](../gui/mcp-server.md) § "The Image Detail view" (two
fields), and
[workspace/workspace.md](../workspace/workspace.md) (the `mipmap_options`
object, and the cache directory in § "Workspace Directory Layout"). Each links back here.

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
A kernel never holds a whole pyramid of an out-of-core photograph: each render
reads a view-region pyramid assembled from photograph tiles, and a kernel that
renders many patches reads each photograph once, in turn, keeping only the
regions its renders need (§ "The caches in memory").

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
| **mipmap** / **mip level** / **mip level file** | a photograph's **mipmap** is its whole stored chain of levels, kept under `.sfm-cache/`: its reduced levels, and its full-resolution copy when the original cannot be read by region. One stored level is a **mip level**, numbered as the pyramid's level is (level 0 full resolution, level `l` reduced by `2^l`), and the file holding it is a **mip level file**, `<file name>-mip<l>-q<quality>.jpg`: `IMG_0001.JPG-mip1-q95.jpg` is half size at quality 95, `-mip2-q95.jpg` a quarter, and `-mip0-q95.jpg` the full-resolution copy; this draft names a file by its level part alone (`-mip0`, `-mip1`) where the quality does not matter. The number after `mip` is the level index the code uses; no factor is converted to a level anywhere. `sfmtool_core::camera::mipmap` | `mipmap set`, a single file called a mipmap, `level file`, `cache file`, `pyramid file`, factor names such as `-mipmap2` | the user chose the standard usage, so others find it familiar: Williams, "Pyramidal Parametrics" (SIGGRAPH 1983), coined *mip map* for the whole prefiltered set of an image at successive halvings; OpenGL says a texture has a mipmap whose images are *mipmap levels*, numbered by level of detail with level 0 the base image (`glTexImage2D`'s `level`); Direct3D and Vulkan say *mip chain* and *mip level* (`mipLevels`); KTX and DDS store a whole chain in one file. *Level* alone stays the in-memory pyramid level, which for a small photograph has no file. Names about the whole feature keep *mipmap* (`mipmap_options`, `sfm ws mipmaps`, `build_mipmaps`, `MipmapError`, `MipmapLocation`, `sfmtool_core::camera::mipmap`), since they act on photographs' mipmaps, plural in the standard sense. The viewer, the CLI and the Python messages use the same words as the code, with no separate user-facing ones: the quiet task *Build mipmaps*, the line *Building mipmaps: {done} of {total}*, a panel's *Building mipmap…* while the one photograph's mipmap it waits for is made, the phase *build mipmaps*, and *mip level* where a message is about one level. A mipmap is made inside the photograph cache's `get` or `get_many`, in the path's slot, with no builder object |
| **region-readable** (an original) | an original JPEG whose layout passes the test in § "Which originals can be read by region", so a rectangle of it decodes to exactly the pixels of the whole decode; its level 0 is the original, and it has no `-mip0` | `tiled`, `seekable`, `RST-capable` | names what the test decides, that a region can be read, rather than one ingredient of it |
| **cache directory** | `.sfm-cache/` at a workspace's root, holding files derived from the workspace's images that can be deleted at any time and rebuilt; today only mip level files | `cache folder`, `.cache` | it names what is in it (files that are a cache) and where (the workspace's); a name without `sfm` would collide with other tools' caches |
| **out-of-core photograph** / **in-core photograph** | a photograph of a workspace whose file has `mipmap_options`, with a width or height over that object's `out_of_core_long_side` (4096 as `sfm ws init` writes it), is out of core (§ "Which photographs are out of core"; one with a side of 65536 px or more is refused everywhere): its reduced levels come from mipmaps, its level 0 from the original or a `-mip0`, and every reader reads it a region at a time; any other photograph is in core: its whole pyramid is built in memory from one decode, as every pyramid is today | `large image`, `tiled image`, `big photograph` | the tier decides where the levels live, which is what *core* means here; *large* says nothing about what is done differently, and every photograph is shown in tiles |
| **photograph tile** | a 512 x 512 block of one level of a photograph's pyramid, with a one-pixel border of its neighbours' pixels to assist bilinear interpolation: the unit the display draws as one texture and that a level is read in; `PhotographTile`, keyed by level, column and row | bare `tile`, `block`, `chunk` | bare *tile* is a view's tile, the patch a kernel renders ([GLOSSARY.md](../GLOSSARY.md) § "Rendering a view's tile"), and Track View draws those beside photographs, so the photograph's blocks need a qualified name |

## Which photographs are out of core

A photograph is **out of core** when its workspace's file has a
`mipmap_options` object (§ "The workspace setting") and the photograph's width
or height, read from its header with `image_dimensions`, is greater than that
object's `out_of_core_long_side` (4096 as `sfm ws init` writes it,
`DEFAULT_OUT_OF_CORE_LONG_SIDE`); it is **in core** otherwise, and always in a
workspace without the object or with no workspace at all.

**A photograph with a side of 65536 px or more is refused.** The photograph
cache refuses any photograph whose level 0 is 65536 px or more wide or high
(over `MAX_JPEG_SIDE`, 65535), in every workspace, with or without
`mipmap_options`, and with no workspace, so the answer does not depend on the
setting: every file the tier may write, a `-mip0` file of a non-region-readable
original included, must fit in a JPEG frame, whose sides are at most 65535,
and one rule for every photograph keeps a photograph's readability the same
whatever the workspace sets. It is decided from the header
(`image_dimensions`), before any decode, and is the error
`GetError::TooLarge { path, width, height }` (`PhotographReadError::TooLarge`
for a kernel), which names the photograph and its size; the Python bindings
raise it as `OSError`, as `read_image_rgb` raises a decoder's limit. Only a
non-JPEG original can be that large. The limit applies to level 0, not to
level 1, because `-mip0` is a level-0 file: a 70000 x 3000 PNG out of core
would need a `-mip0` that cannot be written. The readers outside the photograph cache,
`read_image_rgb` and the commands built on it, SIFT extraction, the display
thumbnails and the image readings at save, are unchanged: they write no mip
level file, so no JPEG frame limits them, and their own limit stays the
reader's 4 GiB decode limit. A photograph with a side of 65536 px or more
that today decodes within that limit is therefore readable by them but
refused by every pyramid reader, in the viewer and in the kernels alike. The
test data (270 x 480 to 2040 x 1536 and 480 x 480),
3840 x 2160 video frames and 4032 x 3024 phone photographs are in core and keep
today's in-memory pyramids, so nothing is written for them and no result
changes except through the pyramid depth (below). A 24-megapixel camera
(6000 x 4000) and the 200-megapixel phone photographs are out of core.

Why 4096 rather than 8192, the texture limit: once every photograph is drawn in
tiles (below), the texture limit no longer separates anything, and the tier is
about memory and the time of a first read. A 6000 x 4000 photograph is 72 MB
at level 0 and 96 MB as a pyramid, so a few hundred of them, a normal camera
capture, do not fit any budget, while their mip level files cost one extra encode each.
4096 is also where `sfm ws init` stops SIFT image doubling
(`double_image_below_long_side`: an image is doubled only when both sides are
below 4096), so at the default SIFT never doubles an out-of-core photograph.
The user has decided that the threshold is a workspace setting: which tier a
photograph is in decides which pixels the kernels read, so a workspace keeps
its own value if the default changes later, as it keeps its feature options,
and a change to it is a change of results that the workspace's owner makes
knowingly. Tests set a small threshold the same way, in the workspace file of
a temporary workspace (§ "The workspace setting").

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

**Mip levels stored.** An out-of-core photograph's mipmap holds a level-0
mip level file, `-mip0`, exactly when its original is not region-readable
(§ "Which originals can be read by region"), and mip level files for levels 1
to `K`,
where `K` is the first level whose longer side is at most one tile (512 px), or
the last level whose sides are both at least 1 px if that comes first. Levels
below `K` are not stored: they are box-filtered in memory from the decoded
level `K`, which is deterministic because its input is the file. So the
coarsest stored level of every out-of-core photograph fits in one photograph
tile.

| Photograph | Levels | Mip level files (`K`) | Coarsest mip level |
|------------|--------|---------------|-----------------|
| 16320 x 12240 | 14 | 5: `-mip1` to `-mip5` | 510 x 382 |
| 6000 x 4000 | 12 | 4: `-mip1` to `-mip4` | 375 x 250 |
| 4097 x 2304 | 12 | 3: `-mip1` to `-mip3` | 512 x 288 |
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

## Which originals can be read by region

Level 0 of an out-of-core photograph is read a tile at a time, by every
reader. An original that allows it is read directly; every other one gets a
a `-mip0` file, a tiled re-encode of full resolution, made once, and from then
on **every reader reads level 0 from `-mip0`, never from the original**:
the display, every kernel, `clipped_share`, the grey reads of *Find Nearby
Tracks*. This is the user's decision. No reader of level 0 holds a whole
decode of an original while it reads regions of it: every reader of level 0
holds only tiles.

**The test.** One function, `region_layout(bytes) -> Result<RegionLayout,
NotRegionReadable>` in `sfmtool_core::camera::mipmap`, reads the original's
bytes and nothing else (not its name, not its extension, not the machine), so
every reader, in every process, decides the same way for the same file. An
original is **region-readable** when all of these hold, checked in order; the
first that fails is the reason recorded in `NotRegionReadable`:

1. It starts with the JPEG start-of-image marker. (The contents decide, as for
   the reader: a PNG named `.jpg` is not a JPEG.) Every non-JPEG format fails
   here: PNG, TIFF, WebP, BMP, OpenEXR and the rest.
2. Its frame is `SOF0` or `SOF1` (baseline or extended sequential, Huffman
   coded) with 8-bit samples and a height that is not 0 (a `DNL` marker is
   not followed). Progressive (`SOF2`), lossless (`SOF3`), arithmetic-coded
   (`SOF9` and up) and 12-bit frames fail.
3. It has one or three components. With three, the colour transform
   `jpeg-decoder` 0.3.2 chooses for the file, and so `ImageU8::read_rgb`
   applies, is YCbCr. That decoder decides by the component identifiers first
   (1, 2, 3 is YCbCr; `R`, `G`, `B` is RGB; two other sets are the big-gamut
   transforms), then by a JFIF segment (YCbCr), then by an Adobe segment's
   transform (1 is YCbCr, 0 is RGB), and takes YCbCr when none of them
   decides (`determine_color_transform` in its `decoder.rs`). Any other
   answer fails, as do CMYK and YCCK.
4. The chroma components are sampled 1 x 1 and the luma `H x V` with `H` and
   `V` each 1 or 2: 4:4:4, 4:2:2, 4:4:0 or 4:2:0. Any other factors fail.
5. It has exactly one scan before end-of-image, holding every component in
   frame order, with spectral selection 0 to 63 and no successive
   approximation.
6. A `DRI` segment before the scan sets a restart interval `R > 0`, and `R`
   divides the MCUs per row, so every interval lies in
   one MCU row and every row starts one. The MCUs per row are
   `ceil(width / (8·H))` with three components; with one, the scan is not
   interleaved and an MCU is one 8 x 8 block whatever the component's
   sampling factors, so they are `ceil(width / 8)`. The rows are counted the
   same way, with `V` and 8. An interval that runs from one row into the next
   fails, because the two-dimensional small frame below cannot be cut from
   it. An interval is also at most 1024 px wide (`R` MCUs of `8·H` px, or of
   8 px with one component), two tiles: a camera that writes one interval
   per MCU row passes the rest of this rule, but then every 512 px tile of a
   16320 px original decodes a strip of the whole width, 32 times the pixels
   of the tile (about 30 to 40 ms a tile instead of about 3, estimated from
   the measured whole decode), so such originals get a `-mip0` instead
   (open question 12).
7. Indexing the scan finds exactly `rows · (MCUs per row / R)` intervals, each
   holding at least one byte, separated by `RST0` to `RST7` in sequence modulo
   8, and the end-of-image marker after the last. A missing, repeated or
   out-of-order marker, a file cut short, or any other marker inside the scan
   fails. Bytes after end-of-image are ignored, as the reader ignores them.

The measured photograph passes (4:2:0, `R = 4` of 1020 MCUs per row, 195,075
intervals), and the other 16 of the user's photographs have the same restart
interval; whether all 17 pass every rule is checked once the test exists
(step 5). A camera JPEG without a `DRI` segment, a progressive JPEG and
every PNG or TIFF fail and get a `-mip0`.

The test is part of the definition of each photograph's pixels: changing it
moves photographs between reading the original and reading `-mip0`, which
changes results, so it changes only with a test showing why. In particular,
the region decode below is exact only because of how `jpeg-decoder` 0.3.2
upsamples chroma and where its SIMD colour conversion stops; a decoder change
that broke the equality test (step 6) would have to change this test to send
the affected layouts to `-mip0`, as a deliberate change of results, rather
than let region reads drift from whole decodes.

**What still decodes an original whole.** Making a photograph's mipmap
(once per mipmap, below); the readers that are not pyramid readers and are
unchanged by this design: SIFT extraction (`build_sift_file`, the Python
extractors, COLMAP), the image readings at save, `read_image_rgb` and
`read_image_rgba` and every command that reads whole arrays through them
(`undistort`, `to-nerfstudio`, optical flow's `load_gray`, the drawing
commands, `xform --add-thumbnails`); and the display thumbnail of an in-core
photograph. No pyramid reader of an out-of-core photograph does. SIFT
therefore reads the original's exact pixels while the kernels read a
non-region-readable photograph's level 0 from `-mip0`, which differs by
0.07 to 0.09 grey levels on average for a JPEG original (measured below); the
keypoints' positions are not affected by a difference that size in any way
this design relies on.

## Mip level files

### Where they are and what they are called

A photograph's mip level files are found from its path alone, so the cache's interface
keeps taking a path. The **workspace root** is the closest ancestor of the path
holding `.sfm-workspace.json`, found by `find_workspace_for_path`
([archive-io-crate.md](../formats/archive-io-crate.md); the walk goes up the
path as given and does not resolve links, so an image folder that links to
storage elsewhere still belongs to the workspace around the link). The
photograph's **relative path** is its path relative to that root. Image names
are normalized paths within the workspace, which the format requires
([sfmr-file-format.md](../formats/sfmr-file-format.md#imagesnamesjsonzst)
§ `images/names.json.zst`), so a photograph's relative path stays inside the
workspace and every photograph of a workspace has a mirror path in
`.sfm-cache/`. The mip level
file of level `l` is

```
<root>/.sfm-cache/<parent of relative path>/<file name>-mip<l>-q<quality>.jpg
```

for example `<ws>/images/IMG_0001.JPG` at the qualities `sfm ws init` writes
has `<ws>/.sfm-cache/images/IMG_0001.JPG-mip1-q95.jpg` to
`<ws>/.sfm-cache/images/IMG_0001.JPG-mip5-q95.jpg`, and an
`IMG_0001.JPG-mip0-q95.jpg` when it needs one. The user has decided this rule (open question 1). It is the rule
a `.sift` file's name follows (`sift_path_for_image` in
[sift/files.rs](../../crates/sfmtool-core/src/features/sift/files.rs) appends
`.sift` to `image_path.file_name()`, so `IMG_0001.JPG` has
`IMG_0001.JPG.sift`). The parts:

- **`<file name>`** is the original's whole file name, extension included, as
  `Path::file_name` gives it and in its own case, as for a `.sift`:
  `IMG_0001.JPG`, `a.b.png`, `README` for a name with no extension.
- **The suffix** is `-mip<l>-q<quality>.jpg`, in lower case, whatever the
  original's format, where `l` is the level index the code uses: the number
  after `mip` equals the level the file holds, so `-mip0` is full resolution
  and `-mip1` half size, and no reader converts a reduction factor to a
  level. `quality` is the JPEG quality the file is encoded at, in decimal
  without leading zeros: `full_resolution_jpeg_quality` for `-mip0`,
  `jpeg_quality` for every other level (§ "The workspace setting").
- **Why the quality is in the name.** A file at another quality is another
  file, so a change of a quality setting is unambiguous: files at the new
  quality have new names and are made as they are read, nothing at the old
  quality becomes stale, and switching the setting back finds the earlier
  files still there and current. Files made with different options never
  mix, as `.sift` files made with different feature options never do, since
  their feature folders are named by a hash of the options
  (`features/sift-sfmtool-<hash>`). The size threshold, the tile size and the
  restart interval are not in the name: none of them changes a file's
  content.
- **The directory parts are the relative path's own**, in their own case, with
  `/` between them.

Two photographs in one directory have different file names, so their mip level
files have different names: the rule is one-to-one, and `x.jpg` beside `x.png` gives
`x.jpg-mip1-q95.jpg` and `x.png-mip1-q95.jpg`. A photograph's name with the
suffix cannot equal another photograph's name with it either, since the suffix
fixes where the original's name ends. Every mip level file still records the relative
path of the photograph it was made from (below), because a cache file can
still be reached for a photograph it was not made from, and the record is how
a reader tells:

- **A cache file copied or renamed by hand**, or left by an image that was
  moved: its `source` names another path, and test 2 of § "When a mip level file is
  current" finds it stale.
- **Two originals whose names differ only in letter case sharing one cache
  file.** An image folder that links to case-sensitive storage (a network
  share, a Linux volume) can hold `X.jpg` and `x.jpg`, while `.sfm-cache/` in
  the workspace sits on a case-insensitive file system, so the two photographs'
  mip level files resolve to the same files. The `source` of that file differs from the
  other photograph's path only in case, so test 2 leaves the decision to
  tests 3 and 4, and test 4 then always compares the original's hash with
  `source_xxh128`, even when its length and modification time both match
  (archive extraction and sync tools can give two files the same time, and
  photographs of one size in an uncompressed format have the same length).
  The hash tells two different originals apart, so the photograph that is not
  the file's source finds it stale and makes its own mipmap over it. The two then replace each other's
  files each time the other is read, which is slow but never serves one
  photograph the other's pixels, and the reader logs a warning naming both
  paths.
- **One original reached by spellings that differ only in case.** On a
  case-insensitive file system `images/IMG_0001.JPG` (a reconstruction's
  image name) and `images/img_0001.jpg` (typed to `sfm ws mipmaps`) are the
  same file. Test 2 again leaves the decision to tests 3 and 4; test 4 hashes
  the original once and finds the hash equal, so the mipmap made under one
  spelling is current for the other and is not made again each time the
  caller changes, and the cache remembers the answer, so the hash runs once
  per cache for each spelling. An exact comparison would find each other's
  record stale and rebuild the mipmap at every change of spelling.

`str::to_lowercase` may disagree with what a file system treats as the same
name (NTFS compares through its own upper-case table, APFS folds case its own
way, `ß` against `SS`, the Turkish dotted `İ`). A disagreement can only cost a
rebuild, never wrong pixels: a pair it calls different is found stale by test
2 and made again, and a pair it calls equal is still checked by tests 3 and 4,
with the hash.

A photograph gets no files, and is read through the in-memory equivalent, when
the cache directory cannot be written.

### What is written in one

A mip level file is a baseline JPEG that any program can open:

| Setting | Value | Why |
|---------|-------|-----|
| encoder | the `jpeg-encoder` crate, as `ImageU8::write` uses | one JPEG encoder in the tree |
| quality | the workspace's `mipmap_options` (§ "The workspace setting"): `jpeg_quality` for levels 1 and coarser, `full_resolution_jpeg_quality` for `-mip0`; both 95 as `sfm ws init` writes them (`DEFAULT_JPEG_QUALITY`) | at level 1 of the measured photograph the mean absolute error is 0.49 grey levels at 95 and 0.73 at 90, for 10.8 MB against 7.3 MB; the kernels read these pixels, so the lower error is worth the bytes. Level 0 is measured separately below |
| chroma | 4:4:4 (`SamplingFactor::F_1_1`) for colour; one component (`ColorType::Luma`) when every pixel of level 0 has R = G = B | a tile can be decoded exactly only without chroma subsampling (below); 4:2:0 would save 14% (9.2 MB against 10.8 MB at level 1) |
| restart interval | 16 MCUs (`MIPMAP_RESTART_INTERVAL_MCUS`), 128 px of one MCU row at 4:4:4 | the unit a tile is decoded in (below); 2.4% larger than no restart markers (10.77 MB against 10.52 MB at level 1, quality 95) |
| Huffman tables | the standard ones, not optimized | the encoder writes optimized tables as one scan per component, which a tile cannot be cut from |
| metadata | the JFIF APP0 header `jpeg-encoder` always writes, at its default density, and one APP15 segment, the record below (`Encoder::add_app_segment`); no EXIF, no orientation | the record is how a reader knows the file is current |

All five reduced mip level files of the measured photograph at quality 95 are
15.2 MB together (10.77, 3.21, 0.90, 0.25, 0.07 MB), half the original's
30.0 MB; at quality 90 they are 10.2 MB.

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
  "out_of_core_long_side": 4096,
  "region_readable_original": true,
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

`source` is the relative path with `/` separators. `jpeg_quality` is the
quality this file was encoded with, from the key that applies to its level;
it must equal the quality in the file's name, and a file whose record and
name disagree is damaged and treated as stale (§ "When a mip level file is
current", test 1).
`region_readable_original` is the test's answer for the original, so a mipmap
made when the original had no `-mip0` is not mixed with one that has. `source_xxh128` is the
XXH128 of the original's bytes, the hash a `.sift` file records as
`image_file_xxh128` (`xxh128_of_file` in
[sift/files.rs](../../crates/sfmtool-core/src/features/sift/files.rs)). The
record lives in each file rather than in a manifest beside them, so a file
cannot be separated from the facts about it, and two writers never edit one
shared file.

### The workspace setting

The user has decided that the workspace file turns the out-of-core tier on and
sets its threshold and the JPEG quality of its mipmaps. `.sfm-workspace.json`
gains one object beside `feature_options`, named the way the file already names
a block of settings for one kind of derived file:

```json
{
  "version": 2,
  "feature_tool": "sfmtool",
  "feature_type": "sift-sfmtool",
  "feature_options": { "...": "..." },
  "feature_prefix_dir": "features/sift-sfmtool-…",
  "mipmap_options": {
    "out_of_core_long_side": 4096,
    "jpeg_quality": 95,
    "full_resolution_jpeg_quality": 95
  }
}
```

- **`out_of_core_long_side`**: a photograph whose width or height is greater
  than this is out of core. A JSON integer of at least 1, with no upper limit
  (a value above every photograph's sides puts them all in core). It decides
  which photographs have mipmaps and so which pixels the kernels read.
- **`jpeg_quality`**: the quality of the reduced mip level files, `-mip1` and
  coarser. **`full_resolution_jpeg_quality`**: the quality of `-mip0`. Two
  keys, because the measurements above put the two trade-offs in different
  places: the reduced levels are small and box-filtered, and level 0 of a
  non-JPEG source takes its first loss at this quality. The name says *full
  resolution* rather than *level 0* because the key is read by a person
  editing the file, who knows the image's resolution and not the pyramid's
  numbering. Each is a JSON integer from 1 to 100, the range `ImageU8::write`
  accepts.
- **Every key is required.** When the object is present it holds all three
  keys and no others. A missing key, an unknown key, a value that is not a
  JSON integer (`95.0`, `"95"`, `true`) or one out of range is an error that
  names the file, the key and the value, raised when the workspace is loaded:
  `RuntimeError` from `load_workspace_config` in Python, and in Rust a
  `WorkspaceConfigError` (new) from the reader the cache uses. Never a panic.
  There are no defaults inside the object, so what a workspace's mipmaps were
  made with is always written in its file.
- **No object means no mipmaps.** A workspace whose file has no
  `mipmap_options` has no out-of-core tier: nothing is written to
  `.sfm-cache/`, and every photograph, whatever its size, is in core and read
  through a whole in-memory pyramid. Every existing workspace is in this state
  until its owner adds the object, so it reads exactly as it does today
  except for the two changes every photograph gets: the pyramid depth by size
  and the tiled display. For a large photograph that means:
  - **The display does not crash.** Image Detail draws tiles cut from the
    in-memory pyramid (step 2) and the 3D background one fitting level of it
    (step 3), so no texture is larger than a tile or the device limit.
  - **Every read holds whole pyramids, as today.** A 200-megapixel photograph
    is decoded whole (0.79 to 1.7 s) into a 762 MiB pyramid in the pyramid
    cache before the panel shows anything; a kernel that reads 17 of them
    holds 12.6 GiB while it runs, and one that reads a few hundred does not
    fit in memory. Track View's requests are admitted within the budget
    (§ "The caches in memory"), so the panels do not decode without end, but
    a track of photographs whose pyramids do not fit the budget shows *Not
    loaded* for the rest. This is the price of not turning the tier on, and
    the viewer says so once per session, the first time it reads a
    photograph over 4096 px in a workspace without the object: *This
    workspace keeps whole photographs in memory. Run `sfm ws mipmaps
    --enable` to read large photographs in tiles.*
- **Photographs with no workspace** are in core too, whatever their size, as
  today: there is no workspace file to turn the tier on and no workspace to
  hold a cache directory.
- **`sfm ws init`** always writes the object with all three values (4096, 95,
  95: `DEFAULT_OUT_OF_CORE_LONG_SIDE` and `DEFAULT_JPEG_QUALITY`), so every new
  workspace has the tier on and records what its mipmaps are made with even if
  the defaults change later. Until the last implementation step it does not
  write it (§ "Implementation steps").
- **Turning the tier on for an existing workspace** is adding the object:
  by hand, or with `sfm ws mipmaps --enable`, which writes it with the defaults
  (or with `--out-of-core-long-side`, `--jpeg-quality` and
  `--full-resolution-jpeg-quality`), refuses a file that already has it, and
  then makes the mipmaps.
- **`sfm ws upgrade`** leaves the object as it finds it, absent or present;
  the user has decided it does not offer to add it (open question 13).
- **Removing the object** turns the tier off: from then on no reader reads
  `.sfm-cache/`, and every photograph is read in core again, with its exact
  levels. The files are not deleted by any reader. `sfm ws mipmaps --prune`
  in a workspace without the object deletes every mip level file there, since none is
  read; with the object it deletes only orphans and stale files.
- **Changing a quality** makes nothing stale. The quality is in each mip
  level file's name, so files at the new quality have new names and are made
  as they are next read, the files at the old quality are left where they are
  (`sfm ws mipmaps --prune` deletes them as orphans), and changing the
  setting back finds them again, current. A mipmap is never mixed across
  qualities, since a reader only ever opens the names of the current
  setting.
- **Changing the threshold** moves the photographs whose long side lies
  between the old and the new value from one tier to the other, and no
  other. Each record stores `out_of_core_long_side`, the value it was made
  under, for diagnosis; a mipmap whose photograph is still out of core stays
  current whatever value it records, because the threshold does not change a
  single byte of it. A photograph that moves **into core** is read from its
  whole in-memory pyramid again, its exact levels; its mip level files are no
  longer read, and are orphans that `--prune` deletes. A photograph that moves **out
  of core** has no files yet, so its mipmap is made when it is next read, as any
  missing mipmap is. Its results change in both directions, which is why the
  value is the workspace's and not the code's.
- **One reader, in Rust.** The object is parsed and validated by
  `read_mipmap_options(workspace_root)` in `sfmtool_core::workspace`, beside
  the rules of `feature_options` (`sfmtool-archive-io` only finds the
  workspace and reads none of its files, and it cannot name a type of
  `sfmtool-core`, which depends on it), and `load_workspace_config` calls
  it through a binding rather than validating in Python. The photograph cache
  reads it once per workspace root and again when the file's modification time
  changes.
- **An invalid object.** A photograph whose workspace file has an invalid
  `mipmap_options` is not read as out of core until the file is fixed: its
  readers report the file's error (Image Detail says *Workspace file error:*
  and the message), since reading it in core or at made-up values would give
  pixels the workspace did not ask for.

**What stays a constant.** The tile size (512, `PHOTOGRAPH_TILE_SIZE`) and
the mip level files' restart interval (16 MCUs) are not in the object: neither changes
any pixel a reader gets (a tile decode equals the whole decode), so no result
depends on them. Tests stay fast without them in the file: a test makes a
temporary workspace whose `mipmap_options` sets `out_of_core_long_side` to 64,
so a 300 x 200 synthetic photograph is out of core, and builds its cache with
`PhotographCacheOptions::tile_size` set to 32, which only tests change, so that
photograph has several levels and many tiles. No test reads or makes a large
photograph.

### How the levels are made

From one decode of the original, held in memory:

1. Decode the original with `ImageU8::read_rgb`, the decoder every reader
   uses.
2. Make levels 1 to `K` by `downsample_2x`, each from the previous **exact**
   level in memory, as `ImageU8Pyramid::from_image` does.
3. Encode each of levels 1 to `K` on its own, coarsest first, with the settings
   above; then, when the original is not region-readable, encode the decode of
   step 1 as `-mip0`, last, because it is the largest (1.4 to 2.1 s for this
   photograph) and nothing coarser waits on it.
4. Write each through a temporary file renamed into place (below).

**The reduced levels come from the exact decode, not from `-mip0`.** The
other way would make level 1 the box filter of what the kernels read at level
0, but no sampler relies on one level being the box filter of the level above
(each reads a level by its own coordinates, and `Anisotropic` blends two levels
that are each a fixed copy), so nothing gains from it. Building from the exact
decode keeps every reduced level at one JPEG loss rather than two, makes a
photograph's reduced levels the same whether or not its original is
region-readable (so a change of the region test, or of
`full_resolution_jpeg_quality`, makes a `-mip0` alone, the latter under a new
name), and lets
`-mip0` be encoded last.

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
after the whole mipmap.

**Determinism.** The same original, decoded by the same build of the same
decoder, gives the same level 0; `downsample_2x` is integer arithmetic; and
`jpeg-encoder` with fixed settings writes the same bytes for the same pixels on
one machine. Whether its AVX2 path and its scalar path write the same bytes is
not established, and neither is whether `jpeg-decoder`'s SSSE3 inverse DCT
and its scalar one agree bit for bit (the decoder picks them at run time). This
does not matter for consistency inside a workspace, because **the files are the
record**: once a mip level file exists, every reader reads it, whatever build wrote it.
A newer build never rewrites a valid mip level file to its own bytes, since that would
change results without any input changing. The originals have the same
property today: a JPEG decoded on two machines can differ by the decoder's
platform paths.

Peak memory while one 200-megapixel photograph is made is about 1 GB: 571 MiB
for the decode, the decoder's own component planes while it decodes (about
290 MiB for 4:2:0 at this size), 190 MiB of reduced levels and 15 MB of encoded
files. This is an estimate from the buffer sizes, not a measurement; step 5
measures it. The time is the decode (0.79 to 1.7 s), the reduction (0.18 to
0.37 s) and the encode of all levels at quality 95 (1.2 and 2.8 s in two runs
under load; about 0.6 s unloaded, estimated from level 1's 0.47 s in the
quietest run and the other levels' third of its pixels), so about 1.6 s
unloaded, estimated. A photograph that needs a `-mip0` adds its encode, 1.4
to 2.1 s, and 33 to 43 MB of encoded bytes held until written.

### The in-memory equivalent

When a photograph's mip level files cannot be written (a read-only cache
directory, a full disk), a reader still needs the pixels the files would hold, at the
qualities the workspace sets, because the
kernels must not read the exact box-filtered levels for one photograph and the
JPEG ones for another. It makes the same files in memory instead: steps 1 to 3
above, then keeps the encoded bytes (up to 58 MB for this photograph, with a
`-mip0`, counted against the tile budget) and decodes tiles from them
instead of writing them. With the same build these are the pixels the files
would have held. It costs the whole making
of the mipmap in every session, and a warning in the log says why the files were
not written. The same path serves the session in which a mipmap is first made: the
reader decodes the bytes it has just encoded rather than reading back the files
it has just written.

### When a mip level file is current

A mip level file is **current** when all of these hold, checked in order, and **stale**
naming the first that fails otherwise:

1. It reads as a JPEG with a record of a known `version`, and the record's
   `level` and `jpeg_quality` equal the level and quality in its name, its frame is
   `width` x `height`, and it has a restart interval.
2. Its `source` is the photograph's relative path, or differs from it only
   in letter case (both lower-cased with `str::to_lowercase` compare equal), in
   which case tests 3 and 4 decide whether it was made from this original,
   and test 4 compares the hash whatever the length and modification time;
   its `level` is the level asked for; and its `width` and `height` are that
   level's size for the original's size.
3. The original's header size equals `source_width` x `source_height`.
4. The original's length equals `source_size` and its modification time equals
   `source_modified_ns`. When the length matches and the time does not (a copy,
   a `git checkout`), or when `source` matched the path only after
   lower-casing (test 2), the original is hashed and compared with
   `source_xxh128`; equal hashes make the file current, and the cache
   remembers the answer, for as long as it lives, for this original and this
   spelling of its path, so it hashes it once per cache and spelling, not on
   every read. A different length is stale without a hash.
5. Its `region_readable_original` equals that of the mipmap's other files. A
   mipmap is never mixed: a file made under another answer is stale, and the
   mipmap is made again. Its quality is not compared with the workspace's
   setting: a reader opens only the names of the current setting, so the
   quality of a file it opens is the setting's by its name, and test 1
   checks that the record agrees.
6. For a photograph whose record says the original is not region-readable,
   its `-mip0` is present and current; for one whose record says it is, no
   `-mip0` is read (one left from an earlier answer is an orphan for
   `--prune`).

The region test runs when a mipmap is made, on the bytes the build decodes, and
its answer is kept in the record. A reader takes the answer from the coarsest
mip level file's record while test 4 holds, since the answer depends only on the
original's bytes and test 4 is what says they have not changed; it does not
index the original again at every session's start (20 to 63 ms and the whole
file read, for this photograph). An original read by region is indexed when
its first level-0 tile is asked for, and a failure of that index (a file
changed without its length or time changing, which test 4's hash would not
have looked at) makes the mipmap stale.

A mip level file that does not read, is cut short, or fails any test is treated as
missing, and the photograph's whole mipmap is made again: making one level costs
the whole decode anyway, and the other levels add only their encodes. It is
never a panic and never an error to the caller unless the original itself
cannot be read. A reader checks the coarsest mip level file when it first needs
the photograph, and each finer one when it first opens it, so a session's start
costs one small file per out-of-core photograph and one `stat` of each
original.

### Writers that meet

Every write goes through a temporary file in the same directory
(`.<file name>.<process id>-<count>.tmp`, where the count is the one
process-wide counter this design keeps, as `ImageU8::write` already does: it
only makes temporary names unique between the threads of one process,
whichever cache they write for, holds no data, and so needs no owner),
renamed over the target, as
`ImageU8::write` already does. Unlike `ImageU8::write`, a writer of a mip level file whose
rename fails does **not** write the target directly: if the target now exists
and is current, another writer has made it, and the write counts as done;
otherwise the photograph falls to the in-memory equivalent, held by that cache
for as long as it lives. The temporary file is removed either way.

Two writers of one photograph through one cache are one: the cache makes each
photograph's mipmap once, and every other asker waits for it. Two caches, in one
process or in two (two viewers, or `sfm sift` and a viewer), can make the same
mipmap at the same time;
each writes the same pixels, and the last rename wins. A reader reads a whole
mip level file into memory when it opens it (10.8 MB for level 1 of this photograph,
up to 43 MB for a `-mip0`) and
keeps those bytes, so a cache sees one version of each file even when another
writer replaces it, and holds a file open only while it reads it. Rust's
standard library opens files on Windows with delete sharing, so an open
mip level file should not block another process's rename over it; the two-process test
in step 5 confirms this on Windows, and the rule above (a failed rename over a
current file counts as done) covers it if it does not.

### Keeping the cache directory out of the way

- On creating `.sfm-cache/`, the writer puts in it a `.gitignore` holding `*`
  and a `CACHEDIR.TAG` with the standard signature line
  (`Signature: 8a477f597d28d172789f06886806bc55`), which backup tools skip.
  No `.gitignore` of the user's is edited.
- **Image discovery must skip it.** `expand_paths` in
  [_filenames.py](../../src/sfmtool/_filenames.py) expands a directory with
  `rglob("*")`, so `sfm sift --extract .` at a workspace root, or a `solve` or
  `pick-images` over it, would take every mip level file for an image. The same holds
  for `_sensor_images` in
  [camrig/from_rig_config.py](../../src/sfmtool/camrig/from_rig_config.py)
  (`rglob("*")`), for `match_pattern_and_padding_misses` in
  [camrig/pattern.py](../../src/sfmtool/camrig/pattern.py), which
  `sfm camrig create` and the `.camrig` resolver match image patterns with,
  and for the Rust walk behind named image lists
  ([named_image_lists/walk.rs](../../crates/sfmtool-core/src/workspace/named_image_lists/walk.rs)),
  where a `**` segment descends into every directory. Each skips any path
  under a directory named `.sfm-cache`. This lands before anything writes the
  directory (step 5).
- **Orphans.** A mip level file whose original is gone, or whose record no longer
  matches, is left where it is by every reader: deleting files is not a
  reader's business, and a stale file is rewritten when its photograph is next
  made. `sfm ws mipmaps --prune` (below) deletes the mip level files whose record names
  a photograph that no longer exists or that they are stale for, those whose
  quality in the name differs from the workspace's current setting for their
  level (orphans of an earlier setting), and temporary files older than an
  hour. Deleting `.sfm-cache/` by hand is always safe.
- `test-data` is never written: every image in it is in core.

### Is it a file format?

The [I/O-shape standard](../STANDARDS.md#every-file-format-has-the-same-io-shape)
is about the formats sfmtool defines, one per format crate. A mip level file is a plain
JPEG with one extra segment, read by its own code in `sfmtool_core::camera`
rather than through `sfmtool.fileio`, and nothing outside sfmtool reads the
record, so it is not a format in that sense: no format crate and no
`read_mipmap` / `write_mipmap` / `verify_mipmap` quartet. The record's fields,
the naming rule and the encoder settings are specified in this design and, once
filed, in a section of the photograph-cache spec, which is the one place that
defines them. If a later tool wants to read mipmaps (the web export, say), the
record becomes a format and gets a spec under `specs/formats/`.

## Reading pixels

### A tile of a mip level file

A photograph tile of level `l ≥ 1` is decoded from the level's mip level file without
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
conversion, which is what the identity was measured against. Only the coarsest
mip level, level `K`, is decoded whole in use: the levels below it are computed
from that decode (§ "Levels by size"). A whole decode of level 1 of this photograph to
RGB took 0.16 to 0.45 s; the raw path does the same work less the colour
conversion, which it then does itself.

### A region of level 0

Every level-0 region of an out-of-core photograph is a tile decode, from one of
two files:

- **A region-readable original** (§ "Which originals can be read by region")
  is read directly. The small JPEG keeps the original's layout in two
  dimensions: its columns are the region's MCU columns widened to interval
  boundaries, its rows are the region's MCU rows, both with a margin, and its
  intervals are copied in raster order. It is decoded to RGB by the same
  `jpeg-decoder` path `read_rgb` uses, then cropped. The margins make the
  cropped pixels equal the whole decode's: one MCU above, below and to the
  left, for the chroma upsampling's neighbours; and on the right at least one
  MCU and at least 16 px, so the cropped pixels are never among the last 8 to
  15 of a line, which the decoder converts with its scalar formula. Where the
  region reaches the image's right or bottom edge, the small frame's width or
  height is set to the distance to that edge, so the edge is padded as in the
  whole decode, and the scalar tail falls on the same pixels in both.

  Measured on the photograph (4:2:0, interval 4 MCUs, one interval per 64 px):
  five 512 px regions, among them the top-left corner and the bottom-right
  corner, 2.6 to 3.5 ms each, reading about 29 KB of the file each, every pixel
  equal to the whole decode. Indexing the original's 195,075 intervals took 20
  to 63 ms.
- **Every other original** is read from its `-mip0`, by the tile decode of
  a mip level file above, and never from the original.

Neither path holds a whole level 0 in memory, so no reader of an out-of-core
photograph's level 0 holds more than the tiles it asked for.

## The caches in memory

`PhotographCache` keeps its job and its key, and holds two kinds of entry.

**Pyramids, for in-core photographs only.** `get`, `peek`, `peek_state` and
`get_many` answer an in-core photograph with an `Arc<ImageU8Pyramid>` of its
whole depth, built as today, under the pyramid budget as today.

**Tiled photographs, for out-of-core ones.** The same calls answer an
out-of-core photograph with a `TiledPhotograph`: its layout, its coarsest
levels and a handle to the cache's tiles, and no other pixels. A kernel never
receives, and the cache never builds, a whole pyramid of an out-of-core
photograph. The kernels read it through view-region pyramids
([view-region-pyramid.md](view-region-pyramid.md)): each render computes from
its warp map the rectangle of each level it reads, and the cache assembles
those rectangles from photograph tiles. The answer is one type for both tiers,
`PhotographSource::{Whole(Arc<ImageU8Pyramid>), Tiled(Arc<TiledPhotograph>)}`,
which is what a kernel's `ProjectedImage` holds, so the kernels' loops do not
branch on the tier. When an out-of-core photograph's mip level files are missing or
stale, the reader that finds it so builds the mipmap itself, as a reader that
finds an in-core photograph not decoded decodes it today (§ "Building a mipmap
in the slot"), so a kernel never reads levels that are not the files' pixels.

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
  there.
- **Rayon workers wait.** A worker that builds or waits on a slot blocks for
  the build, 2 to 4 s at 200 MP against 0.5 to 1.7 s for today's decode of
  the same photograph. This is accepted, as it is for decodes today; the
  panels' tile decodes and layout requests share the pool and are slowed
  while it is busy, not stopped, since a waiting worker holds one thread.
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
  when the maker unwinds. The same slot type serves in-core decodes, which
  gain the same cancellation between the decode and the pyramid. An
  out-of-core photograph goes through its slot whatever the pyramid budget,
  even at 0, where today's `get` decodes in-core photographs without one,
  since a mipmap is stored in files that every asker reads, not memory the budget counts.
- **Progress.** A build is part of the `get` or `get_many` that makes it and
  reports through that caller's `Progress`: a `get` opens phases *decode*,
  *reduce* and *encode* (the last counting levels); `get_many` counts the mipmaps
  it made and the photographs it read in its own phase, under the name its
  caller gives it (*build mipmaps*, *decode images*), with a note
  (*3 of 17 read, 2 mipmaps made*). Nothing on the cache reports on its own:
  there is no reporter, queue or status object.
- **Memory while building.** A build holds a whole decode, about 1 GB at
  200 MP (5 bytes per source pixel, estimated), so `get_many` does not start
  all its misses at once. It first resolves each path's layout (a header
  read, the region test and a record check) to know which are out-of-core
  misses and their sizes, then makes those misses in groups, one
  `par_iter` per group, the groups one after another, with the group's size
  set by memory as `default_concurrent_images_for_dimensions` sizes SIFT's:
  the number of mipmaps that fit in the cache's **build budget**
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
  granting as many of the `wanted` builds as fit under the budget, and returns
  a guard that gives each build's bytes back as that build ends and the rest
  when it is dropped. The builds do not add to the count again. A share
  belongs to the caller that makes a mipmap, not to one that waits: `get`
  reserves its one build when it marks the slot as being made, whether the
  slot was empty or a cancelled maker has just emptied it, and holds nothing
  while it waits; a `get_many` member that finds its slot being made by
  another caller keeps its share while it waits, since it may take the build
  over, and gives it back when it reads the mipmap or stops. A maker that is
  cancelled, fails or unwinds gives its share back through the guard that
  empties its slot. So two
  callers sizing a group at the same moment cannot both take the same free
  bytes: the second sees what the first reserved. Nothing waits on the
  count; it is not a semaphore. A caller that finds the budget used up is
  still granted one build, so that it makes progress, and that one build is
  the only way the count goes past the budget: by at most one mipmap per caller
  reserving at that moment. In-core misses are decoded in parallel as today
  and are not counted: as today, they are bounded by the rayon pool and the
  in-core threshold.
- **What bounds builds from independent callers.** The bound is per cache.
  Every build is reserved through the cache's `reserve_builds` before it
  starts, by `get_many` for its group and
  by `get` for its one mipmap, so the count is always complete, and the count
  goes past the budget by at most one mipmap for each caller that reserved while
  it was full. The number of such callers is bounded because each caller is
  one thread that makes one mipmap at a time unless it is a `get_many`, whose
  group is granted within the budget: a caller that wants several photographs
  asks for them with one `get_many`, never with `get` from many workers of a
  parallel loop (no caller does today; the reference picks call `get` one
  photograph at a time on their job thread). In the viewer the callers that
  can build at once are the two background queues' tasks (one of them perhaps
  the *Build mipmaps* task below) and **at most one panel request**:
  the viewer keeps at most one out-of-core photograph request in flight across
  all its panels (§ "Readers on the GUI thread"), and panels do not read the
  count. All of them use the viewer's one session cache, so the builds in
  progress hold at most the build budget plus three mipmaps: on the 64 GB
  development machine 16 GiB plus 3 GB at 200 MP, and on a 16 GB machine
  4 GiB plus 3 GB. A CLI command uses one cache for its run, and one
  `get_many` runs at a time, so its bound is the budget, or one mipmap when a
  single mipmap is larger than the budget; a Python script that holds one cache
  and calls it from one thread gets the same, and one that calls it from
  several threads at once adds at most one mipmap per thread. A program that
  makes several caches gets one such bound per cache, by its own choice.
  These are on top of the caches' other budgets, and are estimates from the
  buffer sizes.
- **No build on the GUI thread, and no tile task that waits on one.** The
  panels never call `get` on the GUI thread for an out-of-core photograph,
  whatever the budget, and ask for tiles only once a photograph's files are
  current (§ "Readers on the GUI thread"), so a tile task never meets a build.

`get_tile`, `view_region_pyramid` and `get_level` are asked only of a
photograph whose layout is `Known` (its files current), never make a mipmap, and
decode only tiles of current files, milliseconds each (`get_level` at most a
few hundred milliseconds for a level of 48 tiles), so they take no
`&Progress`. A kernel whose files go stale after its read phase found them
current (another process rebuilt them, or the original changed) gets `TileError::Changed` from its next tile read and fails as
§ "Files that change while a kernel runs" says, rather than rebuilding inside
a render and going on with levels of two versions.

**Files that change while a kernel runs.** A tiled source records, for each
file it has read, the file's record. When a tile is decoded again after its
file's bytes were evicted, the file is read again and its record compared
with the one recorded: equal (the same source stamp, quality, region answer,
encoder and sfmtool version, so the same deterministic output) and the kernel
goes on; different, and the read fails with `TileError::Changed`, naming the
file, which the kernel reports as its failure (*The mipmap of
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
| in core | any | cut from the photograph's pyramid entry, which a tile request asks for |
| out of core | 1 to `K` | the level's mip level file, a tile decode from its bytes |
| out of core | below `K` | the levels computed from level `K`, kept whole with the coarsest |
| out of core, region-readable original | 0 | a region decode from the original's bytes |
| out of core, other original | 0 | the `-mip0`, a tile decode from its bytes |

A file's bytes (10.8 MB for level 1 here, 30 to 43 MB for an original or a
`-mip0`) and its restart index are an entry too, read once and kept while
tiles of it are asked for, so a session reads one version of each file.

The tile entries have a budget of their own, so a kernel's in-core pyramids
cannot evict what the display is showing, and the other way round: by default a
sixteenth of physical memory clamped to 512 MiB to 4 GiB, overridden by
`SFM_EXPLORER_TILE_CACHE_MB` as the pyramid budget is by
`SFM_EXPLORER_PHOTOGRAPH_CACHE_MB`. Eviction is least recently used, with one
exception: the coarsest level of every out-of-core photograph whose mip level
files have been read, with the levels below it (at most one tile, 0.78 MB for this
photograph), is evicted only after every other tile entry. A tile or region
held by a running render stays alive until the render drops it, as a pyramid
does today.

**States, and what a caller on the GUI thread may do.** `peek_tile` and
`layout` answer without reading or waiting:

```rust
pub enum PeekedLayout {
    Known(Arc<PyramidLayout>), // header read, files current: tiles may be asked for
    Building { coarsest: Option<Arc<PhotographTile>> }, // mipmap missing, stale or being made; the coarsest once published
    Unreadable(String),         // the original or the workspace file cannot be read; the message says which
    NotKnown,                   // not checked yet
}
pub enum PeekedTile { Ready(Arc<PhotographTile>), Building, Unreadable, NotReady }
```

`Known` is answered only once the photograph's files have been checked current
(or its in-memory equivalent made); `peek_tile` answers `Building` for any tile
of a photograph that is `Building`. A tile is asked for (`get_tile`) only for a
photograph that is `Known`, so a tile request is always a decode of a file
that exists, milliseconds of work, and never waits on a build. If the files
change under a `Known` photograph, `get_tile` finds the record stale, answers
`TileError::Changed` without waiting, and the photograph goes back to
`Building`, so the panel requests it again. A mipmap is made only inside a
`get` or `get_many`, never on the GUI thread.

**Memory, under this design.** What a kernel holds for out-of-core photographs
is the view-region pyramids of the renders it has in flight or keeps for reuse,
plus the tile budget, whatever the number or size of the photographs. A
render's region pyramid at patch resolution `R` with a footprint that is not
stretched is about `(R + 4)^2 x 3` bytes at its level and a quarter of that at
the next (anisotropic blending), so 2.9 KB at `R = 24` and 17 KB at `R = 64`,
and 2.25 times that when it is kept for reuse with its margin (a quarter of
the region on every side). A stretched footprint, a grazing view, or the
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
520 MB, on top of the tile budget; the viewer, with at most two kernel calls at
once (one per background queue) and Track View's renders, at most
2 x 99 MB + 3.1 MB + 35 x 12.7 MB, about 650 MB. A build renders nothing
and pins no tiles, so building mipmaps on these threads does not change the
count; a build's own memory is the build budget's bound in § "Building a mipmap
in the slot", separate from this one. The common case is a few hundred
kilobytes, whatever the number of views or photographs (estimates from the
sizes).

| Case | Today (whole pyramids) | This design |
|------|------------------------|-------------|
| 17 x 200 MP, *Find matches by geometry* over the views the cull keeps (up to 151), `R = 24` | up to every photograph, 12.6 GiB for all 17 | up to 151 x 2.9 KB x 2.25 = 1.0 MB of regions kept for reuse (well inside the call's 99 MB pool), plus at most the tile budget (4 GiB at 64 GB of memory, 512 MiB at 8 GB) |
| 17 x 200 MP, display patch bitmaps at open, one render per point | 12.6 GiB | 32 renders in flight x 2.9 KB, plus the tile budget |
| 500 x 200 MP, any of the above | 372 GiB; fails | the same as for 17: regions plus the tile budget |
| Any of the above, worst case (the kernel call's region pool full and every rendering thread on a grazing view) | — | one kernel call: 99 MB + 33 x 12.7 MB, about 520 MB; the viewer with two kernel calls and Track View: about 650 MB; on top of the tile budget |
| 500 x 6000 x 4000, any of the above | 500 x 96 MB = 48 GB | the same |
| 17 x 200 MP, Track View on a 17-image track at `R = 64` | 12.6 GiB | 17 x 17 KB x 2.25 of regions, plus the tiles they are cut from, one to four per level read and one or two levels per view, 13 to 108 MB |
| 368 x 3840 x 2160 (in core) | 12.2 GB held by a `get_many` over every frame | unchanged: in-core photographs keep whole pyramids |

The cost moves from memory to tile decodes: a render whose tiles are not
cached costs 2.3 to 6 ms per tile it touches, one to four per level it reads.
A bulk kernel whose tiles do not fit the budget decodes some tiles more than
once; it takes its renders in an order that keeps them together (grouped by
image and position where the kernel's structure allows, § "Bulk kernels" in
the companion draft), and its phase notes report the tiles decoded and reused,
so a run that decodes far more than once per tile shows it.

The display holds, for 17 photographs of 200 MP, the visible tiles of one
level (at about one texel per screen pixel a 3840 x 2160 panel covers at most
9 x 6 = 54 tiles, 43 MB) plus the coarsest resident of each photograph
(13 MB). For 368 frames of 3840 x 2160 it cuts tiles from their pyramids and
keeps the coarsest level of each frame it has shown, 480 x 270, 389 KB, 143 MB
for all 368.

**Track View's requests.** Track View's photographs that are in core are
asked for as whole pyramids, and the decode-and-evict loop above is real for
them when a track's pyramids exceed the pyramid budget. Those requests made in
one frame are admitted in the order they are made only while the estimated
bytes of the pyramids they ask for (from header dimensions, levels counted) fit
within 80% of the pyramid budget, counting the pyramids already in the cache
that the frame draws from; the photographs that do not fit are not asked for,
and their tiles show *Not loaded: photograph cache is full* with a hover naming
`SFM_EXPLORER_PHOTOGRAPH_CACHE_MB`. Out-of-core photographs are asked for as
tiles and do not count against it.

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
- **In the viewer**, when a reconstruction opens, the viewer starts a quiet
  task, *Build mipmaps* (`Operation::BUILD_MIPMAPS`, a new quiet
  source), on the fast queue. It is an ordinary background task with its own
  `Progress` and collector: its job calls `get_many` over the
  reconstruction's out-of-core photographs whose files are missing or stale
  (found by a `stat`, the original's header and markers for the region test,
  and one small read each), the selected image first, then the images the
  Image Browser strip shows, then the rest in image order, and the order
  decides which group each falls in. Being a quiet task, it gives way to an
  asked-for task in its queue: its cancel flag is set, each build in progress
  stops within one stage and leaves its slot empty, and the task is queued
  again behind the one that displaced it, losing at most the stage each build
  was in. While it runs, the Background Tasks panel's quiet line shows it as
  it shows every quiet task, as *Building mipmaps: {done} of {total}*. A panel's request or a
  bench step that needs a photograph meanwhile asks the cache for it
  directly; if the task is making it, the slot makes the asker wait for that
  build rather than start a second, and if not, the asker makes it. There is
  no queue across callers.
- **A kernel** that needs an out-of-core photograph whose files are not
  current makes the mipmap inside its read phase (its `get_many`), counted in the
  phase's note (*3 of 17 read, 2 mipmaps made*). It does not read the exact
  levels in the meantime.

**Readers on the GUI thread.** None of them makes a mipmap, decodes a whole
photograph or waits. Each asks `layout` and acts on the state. *Request the
photograph* means what `display_photograph` does today for a decode: once per
path, it starts one rayon task that calls `get`, which makes the mipmap (or waits
for the caller already making it), records the path in `PhotographRequests`
so later frames start no second task, and repaints when the task ends. For an
out-of-core photograph whose files are not current, **at most one such
request is in flight across the whole viewer**: a panel whose request cannot
start because another is in flight keeps it wanted and tries again on a later
frame, so twenty Track View cells that need mipmaps start them one after
another, not twenty at once. Panels do not read the build count; the
request's `get` reserves its one mipmap through the session cache's `reserve_builds` like any other
caller, so the count stays complete. A request for a photograph whose files
are current only reads a layout and tiles, and is not limited.

| Reader | `NotKnown` | `Building` | `Known` |
|--------|------------|------------|---------|
| Image Detail | asks for the layout once (a header read, the region test and a record check, on the rayon pool) and draws *Loading image…* | requests the photograph once, draws the coarsest level if the build has published it and *Building mipmap…* otherwise | peeks the visible tiles and asks for the missing ones |
| Track View's patch tiles, cluster-stage tiles, context and crops; the Track Table's crop column and hover | the same layout request; the cell draws its loading state | requests the photograph once; the cell says *Building mipmap…* | computes the render's regions, peeks their tiles, asks for the missing ones, and renders when all are present |
| The 3D background | the same layout request; nothing drawn | requests the photograph once; draws the coarsest level if published | asks for its level with `get_level`, which assembles it from tiles on the rayon pool, and draws the coarsest until it lands |
| Display thumbnails | built in the open job, off the GUI thread: a photograph whose coarsest mip level file is not current gets the grey placeholder row | the row stays the placeholder | a placeholder row whose photograph has become `Known` is filled, on the rayon pool, from the coarsest mip level file |

A build publishes each reduced level to the slot's `Building` state as it is
encoded, coarsest first, so a panel shows a photograph's coarsest level about a
second after the build starts, before its finer levels and its `-mip0` are
written.

**A budget of 0.** `display_photograph` with `SFM_EXPLORER_PHOTOGRAPH_CACHE_MB=0`
calls `get` on the GUI thread today, since nothing it decodes would be kept for
the next frame. That stays for in-core photographs only. An out-of-core
photograph never takes that path: it is requested on the rayon pool as above
whatever the budget, so a build never runs on the GUI thread; and with a tile
budget of 0, a tile decoded on the rayon pool is handed to the panel that asked
for it, whose texture keeps it, and the cache keeps nothing.

## Display

### Image Detail

The panel stops making a texture of the photograph and draws photograph tiles:

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
  coarsest resident level is drawn at once if its mipmap or pyramid has been
  read this session, then its tiles at the current view. While animation plays,
  the next image's tiles for the current view are asked for one frame ahead.
  The view persistence rule (§ "View persistence across image, reconstruction
  and panel changes") is unchanged; it computes `display_size` from the
  photograph's level-0 size, which the panel reads from the header (or the
  camera) rather than from a texture.
- **Zoom limit.** The limit becomes `max(32, 8 / fit_scale)` times the fit,
  where `fit_scale` is the screen pixels per level-0 pixel at the fit, so a
  level-0 pixel can always be enlarged to 8 screen pixels (open question 7).
  Today's `MAX_ZOOM` of 32 times the fit reaches only 0.78 screen pixels per
  level-0 pixel for a 16320 px photograph in a 400 px panel, so the
  original's pixels could not be enlarged at all; for a photograph that fits
  a panel at 1/4 or larger, the limit stays 32.
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
  out-of-core one the level assembled from its tiles (`get_level`), once the
  photograph is `Known`. It is asked for through the cache on the rayon pool,
  and the coarsest level is drawn until it lands. A photograph that cannot be read is remembered as
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
"Regions" means that for an out-of-core photograph the reader reads
view-region pyramids assembled from photograph tiles: levels 1 to `K` from the
mip level files, level 0 from the original if it is region-readable and from
`-mip0` otherwise. "Depth" means the change of pyramid depth above. Every
row marked "regions" must go through the view-region pyramid before
`sfm ws init` writes `mipmap_options` (step 15): it is the only way those
readers get pixels of an out-of-core photograph, since the cache builds no
whole pyramid of one. Until a reader has moved (steps 9 to 12 and 14), it
refuses a tiled source with an error, which only a test workspace, or one
whose owner wrote the object by hand, can reach. In a
workspace without `mipmap_options`, every row reads as its in-core column
says, for photographs of every size.

| # | Reader | Today | After | Results change |
|---|--------|-------|-------|----------------|
| 1 | Image Detail (`dock.rs` → `display_photograph` → `load_image`) | level 0 as one texture | photograph tiles of the zoom's level; coarsest resident | display only; no texture over 514 px |
| 2 | 3D camera-view background (`bg_image.rs` `load_bg_texture`, from `app.rs`) | whole RGBA, one texture, decoded on the GUI thread | one level fitting viewport and limit, off the GUI thread, through the cache | display only; the alpha it reads today is not drawn |
| 3 | Track View's requests (`dock.rs` `cache_bench_track_images`, `cache_selected_point_images`) | a pyramid request per image per frame | in core: the same, admitted within the budget; out of core: the layout, then the tiles each render needs | no |
| 4–6 | Track View's patch tiles, cluster-stage tiles, hover context (`track_view/body/mod.rs` `ensure_tile`, `ensure_context`; `tile.rs`, `patch.rs` → `render_tile`) | whole pyramid, `peek` | in core: whole pyramid; out of core: regions, from peeked tiles | out of core: the stored levels; depth |
| 7–9 | Track View's crops, the Track Table's crop column and hover (`patch_pictures/crop.rs`, `track_table_panel.rs` `crop_picture`) | one level of the pyramid | the same level; out of core: the crop's rectangle of it, from peeked tiles | out of core: the stored levels; depth past a step of 64 |
| 10 | Track Table reference picks (`track_table_panel/picks.rs` → `render_reference` → `render_view_tile`, `clipped_share`) | `get`, whole pyramid; level 0 inside the outline | in core: the same; out of core: regions, and `clipped_share` walks the level-0 tiles under the outline; each pick still fetches its views with one `get` per observation, and makes one region pool for its one `render_reference` call over all of them, so a point's views share one pool | out of core: the stored levels, and `clipped_share` reads `-mip0` for an original that is not region-readable |
| 11–13 | Display patch bitmaps at open, conversion to embedded patches, `sfm web-export` (`display_bitmaps.rs`, `state/edits.rs` `convert_to_embedded_patches_job`, `web_export.rs`) | `get_many`, 6 levels, every photograph held at once; web-export with budget 0 | `get_many` answers sources; regions for out of core; web-export makes missing mipmaps in the workspace | out of core: the stored levels; depth |
| 14–18 | The bench's steps and searches, live evaluation, *Find Nearby Tracks*, *Create Track Here*, *Add image to tracks*, *Refine all normals*, *Refit all points* (`bench.rs`, `state/edits.rs` `fetch`, `bench/nearby_tracks.rs`, `bench/track_at_pixel.rs`, `add_image_to_tracks.rs`, `state/edits/refine_all_normals.rs`, `refit_all_points.rs`) | `get_many`, levels per sample or per grid, `clipped_share` on level 0 | regions for out of core, `clipped_share` as in row 10 | out of core: the stored levels; depth |
| 16a | *Find Nearby Tracks*' grey reads (`GreyImages`, `bench/nearby/grey.rs`) | level 0 of each view, blurred grey `f32`, whole: 800 MB per 200-megapixel view | in core: unchanged; out of core: each read blurs the level-0 region it samples, plus the blur's 4 px | out of core: level 0 from `-mip0` for an original that is not region-readable |
| 19 | The viewer's cluster-patches build (`cluster_patches.rs` `read_photographs`) | `read_rgb`, full depth, outside the cache, all at once | `get_many` through the cache; regions for out of core | out of core: the stored levels |
| 20 | Display thumbnails when none is stored (`display_thumbnails.rs` → `display_thumbnail_row`) | whole decode, resized to 128 | in core unchanged; out of core from the coarsest mip level file | display only |
| 21 | `.sift` building (`features/sift/files.rs` `build_sift_file`) | whole decode | whole decode of the original, then the mipmap from it | no |
| 22 | Image readings at save (`reconstruction/image_readings.rs` `gather_image_readings`) | whole decode of every photograph with no `.sift` row, in a `into_par_iter` over all images, so one 571 MiB decode per rayon thread at 200 MP | the original, decoded whole, with the number decoded at once bounded by memory as SIFT's is (step 4) | no |
| P1 | `sfmtool.fileio.read_image_rgb` / `read_image_rgba` (`sfmtool-py/src/fileio/image.rs`) | whole decode | unchanged: they read files, not pyramids | no |
| P2 | `ImagePyramidSet(recon, images)` and the kernels' `images` lists (`sfmtool-py/src/patches/views.rs`) | exact pyramids of the arrays passed | unchanged; a new `ImagePyramidSet.read(recon)` reads by path through the core cache, as sources | no for arrays; through `read`, as rows 11–18 |
| P3 | `embed-patches`, the strip montages, the `xform` steps that read photographs (`src/sfmtool/_embed_patches.py`, `src/sfmtool/strips/`, `src/sfmtool/xform/_images.py`) | `read_workspace_image` arrays into the kernels | `ImagePyramidSet.read(recon)` | out of core: the stored levels, so they match the viewer |
| P4 | `sfm cluster-patches` (`src/sfmtool/_cluster_patches.py`) | `read_image_rgb` arrays, full depth | through the same reader as #19 | out of core: the stored levels |
| P5 | The view-tile binding (`sfmtool-py/src/patches/view_tile.rs`) | an exact pyramid of the array passed | unchanged | no |
| P6 | Optical flow's pyramids (`sfmtool-py/src/flow/warp.rs`) | a grey pyramid of its own from an array | unchanged; flow reads level 0 from its array | no |

The measurement harnesses under `scripts/` that build an `ImagePyramidSet` from
`read_image_rgb` arrays (`track_at_pixel/context.py`,
`add_image_to_tracks/harness.py` and `misses.py`) are P2 callers: they read
exact pyramids until they move to `ImagePyramidSet.read`, which step 14 does.
A P2 caller that passes arrays of an out-of-core photograph holds them whole,
as today; that is the caller's choice of input, and `read` is the bounded way.

Every reader that reads a pyramid of an out-of-core photograph now reads the
same levels whether it runs in the viewer or on the command line, and in any
session. A reader of level 0 reads the original's own pixels when the original
is region-readable, and `-mip0`'s otherwise.

## Standards

- **Progress.** `build_mipmaps` takes `&Progress`, opens phases for the
  decode, the reduction and the encodes, counts levels and polls for
  cancellation between them; a cancelled build writes nothing further and
  leaves no partial file. `PhotographCache::get` gains a `&Progress`
  parameter, since a `get` can now make a mipmap (2 to 4 s) and already can take
  over a second to decode a large in-core photograph; callers with nothing to
  report pass `&Progress::none()`. `get_many` passes its own down. `get_tile`,
  `view_region_pyramid` and `get_level` take none, since they never make a mipmap
  and take milliseconds. `sfm ws mipmaps` reports through a
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
  a cache whose tile size is 32 (§ "The workspace setting"). No test reads or
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
pub const CACHE_DIR_NAME: &str = ".sfm-cache";

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
    InMemory(NoFilesReason), // cache directory not writable
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

/// Gives each granted build's bytes back to its cache's count as it ends,
/// the rest on drop. Made by `PhotographCache::reserve_builds`.
pub struct BuildReservation<'a> { /* … */ }
impl BuildReservation<'_> {
    pub fn granted(&self) -> usize;
    pub fn release_one(&self);
}
/// A quarter of physical memory, or the override in the environment: the
/// default build budget. A pure function, like `default_budget_bytes`; it
/// keeps no state.
pub fn default_build_budget_bytes() -> u64;

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
    pub budget_bytes: u64,          // default_budget_bytes(): in-core pyramids
    pub tile_budget_bytes: u64,     // default_tile_budget_bytes(): tiles, file bytes, coarsest levels
    pub build_budget_bytes: u64,    // default_build_budget_bytes(): builds in progress on this cache
    pub tile_size: u32,             // PHOTOGRAPH_TILE_SIZE; only tests change it
}

/// What a kernel reads a photograph through.
pub enum PhotographSource {
    Whole(Arc<ImageU8Pyramid>),   // in core
    Tiled(Arc<TiledPhotograph>),  // out of core: read by view-region pyramids
}

impl PhotographCache {
    pub fn new(options: PhotographCacheOptions) -> Self; // no `levels`: depth is by size
    /// Makes the mipmap of an out-of-core photograph whose files are missing or
    /// stale, in the path's slot on the calling thread, reserving that one
    /// build with `reserve_builds`, or waits for the caller already making it;
    /// reports through `progress`.
    pub fn get(&self, path: &Path, progress: &Progress<'_>) -> Result<PhotographSource, GetError>;
    /// Reserve the estimated bytes of up to `wanted` builds against this
    /// cache's build budget with one compare-exchange loop on its count, at
    /// the moment the caller decides; never waits. Grants as many as fit, and
    /// one when none fit, so a caller always makes progress.
    pub fn reserve_builds(&self, bytes_per_build: u64, wanted: usize) -> BuildReservation<'_>;
    /// The header, the tier, the level sizes and, out of core, the region
    /// answer; makes no mipmap. The blocking form of `layout`.
    pub fn read_layout(&self, path: &Path) -> Result<PhotographLayout, GetError>;
    /// Pixels `rect` of `level` as every kernel samples them, making the mipmap
    /// first if its files are missing or stale (`get`), then cutting from the
    /// pyramid in core or assembling from tiles out of core. A level or
    /// rectangle outside the photograph, or an empty rectangle, is
    /// `OutOfRange`. Behind the Python class's `read_level`, `read_tile` and
    /// `read_region`.
    pub fn read_region(&self, path: &Path, level: usize, rect: [u32; 4], progress: &Progress<'_>)
        -> Result<ImageU8, PhotographReadError>;
    /// Makes its misses in parallel on the rayon pool, the out-of-core ones in
    /// groups sized by the build budget; the return type is today's.
    pub fn get_many(&self, paths: &[&Path], progress: &Progress<'_>)
        -> Result<(Vec<Option<PhotographSource>>, GetManyTally), Cancelled>;
    // peek and peek_state answer `PhotographSource` too; insert, forget and clear
    // as today; `stats` adds the tile entries, bytes, budget, hits and misses,
    // the build budget, the bytes reserved for builds, and the mipmaps made by
    // this cache

    pub fn layout(&self, path: &Path) -> PeekedLayout;
    pub fn request_layout(&self, path: &Path);   // a header read, the region test, a record check
    pub fn peek_tile(&self, path: &Path, key: TileKey) -> PeekedTile;
    /// Only for a photograph whose layout is `Known`; never makes a mipmap, and
    /// answers `Changed` rather than waiting when its files have gone stale.
    pub fn get_tile(&self, path: &Path, key: TileKey) -> Result<Arc<PhotographTile>, TileError>;
    /// A view-region pyramid of an out-of-core photograph, assembled from
    /// tiles ([view-region-pyramid.md](view-region-pyramid.md)).
    pub fn view_region_pyramid(&self, path: &Path, regions: &LevelRegions)
        -> Result<ImageU8Pyramid, TileError>;
    /// One level whole, for the 3D background, assembled from tiles out of core.
    pub fn get_level(&self, path: &Path, level: usize) -> Result<Arc<ImageU8>, TileError>;
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
pub enum TileError { Changed { path: PathBuf }, Unreadable { path: PathBuf, why: String } }
pub enum GetError { NotFound(PathBuf), Unreadable(String), Workspace(WorkspaceConfigError),
    /// A side of 65536 px or more, read from the header; in every workspace.
    TooLarge { path: PathBuf, width: u32, height: u32 }, Cancelled }

/// `read_layout`'s answer: the layout, and `None` in core or whether the
/// original is region-readable out of core.
pub struct PhotographLayout { pub layout: Arc<PyramidLayout>, pub region_readable: Option<bool> }
```

`DISPLAY_PYRAMID_LEVELS` and the viewer's `state::PYRAMID_LEVELS` are removed.
`ProjectedImage`'s `pyramid` field becomes a `PhotographSource`.

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
    /// A reader that has not yet moved to view-region pyramids was handed a
    /// tiled source; exists from step 8 until the last reader moves (step 14).
    NotYetTiled { reader: &'static str },
}
impl From<Cancelled> for PhotographReadError { /* … */ }
impl From<TileError> for PhotographReadError { /* … */ }
impl From<GetError> for PhotographReadError { /* … */ }
```

**In-memory pyramids stay infallible.** `render_tile`, `render_tile_with_grad_into`,
`render_tiles` and the `remap_*` functions keep their signatures and take
`&ImageU8Pyramid`: an in-memory pyramid cannot fail, and the callers that hold
only those (the Python bindings given arrays, the view-tile binding, tests)
are not made to handle an error that cannot happen. Beside them,
`render_source_tile` and `render_source_tile_with_grad_into` take a
`&ProjectedImage` (and so a `PhotographSource`) and the region pool the
kernel's caller made for this kernel call (`RegionPool::for_kernel_call()`,
whatever way the caller fetched its sources; a kernel that calls another
passes its own pool down), and
return `Result<_, PhotographReadError>`: for `Whole` they call the infallible
functions and answer `Ok`, for `Tiled` they assemble the view-region pyramid
or sample through tiles, either of which can fail.

**Kernels pass the error up.** Every kernel that renders from a
`ProjectedImage` calls the `render_source_*` functions and returns the error
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
| 8 | `ViewSources::fetch` in [state/edits.rs](../../crates/sfm-explorer/src/state/edits.rs) and the public `render_patch_bitmap_column` / `render_display_patch_bitmaps` in [display_bitmaps.rs](../../crates/sfmtool-core/src/patch/display_bitmaps.rs) return `PhotographReadError` instead of `Cancelled`; `get_many` still returns `Cancelled`, which converts, and the new variants are `NotYetTiled`, which their kernels return for a `Tiled` source until they move, and the `Unreadable` and `Changed` of a tiled read from then on. Their kernels still render infallibly from `Whole` sources |
| 9 | Track View's and the Track Table's renders, which draw a cell's error state instead of the tile |
| 10 | the bench kernels (`bench::evaluate`, `evaluate_rendering_bitmap`, `score_bitmap`, `fit`, `render_bitmap_in_place`, `search_geometry`, `set_stage`, `build_track_at_pixel`, `classify`) and the patch kernels they and the reference picks share (`view_selection::select_patch_views`, `select_patch_cloud_views`, `reference_view::render_view_tile`, `read_track`, `stored_bitmap::render_reference`, `keypoint_localize`'s and `keypoint_subpixel`'s entry points, `member_coherence`'s, `observation_reading::read_cloud_observations`, `spawn::spawn_candidate_tracks`) |
| 11 | `stored_bitmap::render_patch_bitmap`, `render_patch_cloud_bitmaps`, `refine_normals` and the normal kernels it runs (`pairwise_flow_normal`, `grown_plane_normal`, `grid_plane_normal`, `finite_difference_normal`), `refit_points`, and the cluster refinement's entry points with `sample_member_grid` |
| 12 | `find_nearby_tracks` and the reads under it (`GreyImages::get`, `read_patch_along_ray`, `far_field_sweep`, `depth_layers`, `guided_matches`, `nearby_cluster_tracks`, `constellation_seeds`, `nearby_points`, `distance_range`) |
| 14 | `ImagePyramidSet.read`, the first binding whose sources can be tiled; `NotYetTiled` is deleted. The bindings over the kernels above change with their kernels in steps 10 to 12 and raise the error from then on, though the `Whole` sources they build from arrays cannot fail |

The viewer's side, in
[photograph_requests.rs](../../crates/sfm-explorer/src/state/photograph_requests.rs):
`display_photograph` keeps its contract for in-core photographs; a
`display_tile` beside it acts on the layout state as § "Readers on the GUI
thread" says, starting one `get_tile` per missing tile of a `Known`
photograph; and the admission rule for in-core pyramid requests lives there.
The *Build mipmaps* quiet task is `state/build_mipmaps.rs`: a job
that calls `get_many` over the reconstruction's out-of-core photographs in the
order § "Who makes the mipmaps, and when" gives.

## Python bindings

The user has decided that Python gets the same machinery as the viewer. All
of it is in `sfmtool-core`; the bindings are thin wrappers over it, with no
logic in Python ([STANDARDS.md](../STANDARDS.md#logic-lives-in-rust-not-in-python-wrappers)).
**Only these parts are the viewer's own**: drawing photograph tiles as GPU
textures, the rule of one out-of-core photograph request in flight across the
panels, the *Build mipmaps* quiet task at open, and the admission rule
for in-core pyramid requests in `photograph_requests.rs`. Everything else in
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
which mip level files are not (§ "Is it a file format?"), nor `sfmtool.patches`,
whose names are the patch kernels'.

```python
class PhotographCache:
    def __init__(self, budget_bytes: int | None = None,
                 tile_budget_bytes: int | None = None,
                 build_budget_bytes: int | None = None): ...
    def layout(self, path) -> PhotographLayout: ...
    def read_level(self, path, level: int, *, progress=None) -> np.ndarray: ...
    def read_tile(self, path, level: int, col: int, row: int, *, progress=None) -> np.ndarray: ...
    def read_region(self, path, level: int, x0: int, y0: int, x1: int, y1: int,
                    *, progress=None) -> np.ndarray: ...
    def build_mipmaps(self, paths, *, progress=None) -> dict: ...
    def stats(self) -> dict: ...
```

- **`PhotographCache(budget_bytes=None, tile_budget_bytes=None,
  build_budget_bytes=None)`** holds one core `PhotographCache`; an omitted
  budget is the core default (`default_budget_bytes()`,
  `default_tile_budget_bytes()`, `default_build_budget_bytes()`), and a
  pyramid or tile budget of 0 keeps nothing, as in the core. Every call on one
  `PhotographCache` shares its build count, so a script that holds one cache
  gets the same bound on builds in progress as the viewer (§ "Building a mipmap
  in the slot"). The tile size is not an argument: it is a
  constant (§ "The workspace setting").
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
  `sfm ws mipmaps` expands them, skipping `.sfm-cache`) through the cache's
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
  `bytes`, `budget_bytes`, `hits` and `misses` (today's `stats`, for the
  pyramids), `tile_entries`, `tile_bytes`, `tile_budget_bytes`, `tile_hits`
  and `tile_misses`, `build_budget_bytes` and `build_reserved_bytes` (the
  cache's build count), and `mipmaps_made`. `PhotographLayout` stays a frozen
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
its own short-lived cache, of pyramid budget 0, as before, and drops it when
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

`ImagePyramidSet.read` reads its photographs as sources: whole pyramids for
in-core photographs and tiled ones, read by view-region pyramids, for
out-of-core ones, so each is read the way the viewer reads it, making missing
mipmaps on the way, and the set holds no whole pyramid of an out-of-core
photograph. The `images` argument of the constructor and of every kernel
keeps meaning exact pyramids of the arrays given.

`load_workspace_config` validates `mipmap_options` through a binding over
`read_mipmap_options`, in `sfmtool.workspace`.

## Implementation steps

Each is one pull request, useful on its own, with its tests. Steps 1 to 3 end
the crash with in-memory pyramids, before any file is written, and step 4
bounds the one other whole-photograph reader that runs unbounded. Steps 5 to
8 build the tier without turning it on anywhere but in tests: until step 15,
neither `sfm ws init` nor any other command writes `mipmap_options`, so no
workspace has the object unless its owner writes it by hand, and every
photograph stays in core. Steps 9 to 12 and 14 move the readers to view-region
pyramids, one group per pull request; a reader that has not moved refuses a
tiled source with an error, which only a test workspace, or one whose owner
wrote the object by hand, reaches. Step 15 turns the tier on.

1. **Depth by size.** Pyramids of `full_levels` depth everywhere;
   `PhotographCache::new` loses `levels`; `DISPLAY_PYRAMID_LEVELS` goes.
   Tests: the depth for several sizes; a kernel render that clamped at level 5
   now reads level 6, on a synthetic view shrunk 64 times; the tests that pin
   rendered values updated where they change, each change explained.
2. **Image Detail draws photograph tiles** cut from the in-memory pyramid,
   with the level rule, the coarser fallback, the per-frame upload cap, the
   texture limit of 256 and the zoom limit `max(32, 8 / fit_scale)`. This
   alone ends the crash in Image Detail for every
   photograph, though a 200-megapixel one is still read whole until the tier
   is on for its workspace. Tests (headless, the panel's lib tests): the level
   for a set of scales; the zoom limit for a large and a small photograph
   (8 screen pixels per level-0 pixel reached, and 32 kept); the tiles a view
   asks for; a 9000 x 16 synthetic
   photograph never makes a texture over 514 px; the fallback draws the
   coarser level where a tile is missing; the test data's photographs look as
   they did.
3. **The 3D background** from one level by the viewport and the limit, off the
   GUI thread, through the cache, with a remembered failure. Ends the crash in
   camera view. Tests: the level rule; the texture coordinates' scale for odd
   sizes; an unreadable photograph is read once.
4. **Image readings at save, bounded.** `gather_image_readings` decodes the
   photographs that have no `.sift` row in an `into_par_iter` over every image,
   so at 200 MP each rayon thread holds a 571 MiB decode and its grey copy at
   once. Its decodes run on a pool sized by memory the way SIFT's are since
   #1306 (`default_concurrent_images_for_dimensions`, from the photographs'
   header sizes), and the rows that need no decode stay parallel. Tests: the
   concurrency for a set of sizes and memory figures; the rows are identical.
5. **Mip level files.** Image discovery skips `.sfm-cache` (`expand_paths`, the
   camrig and rig-config finders, `match_pattern_and_padding_misses`, the
   named-image-list walk) with a test each; then `PyramidLayout`, the region
   test with its interval-width rule, `mipmap_location`, the record,
   `MipmapOptions` and `read_mipmap_options` with their validation (but not
   yet `sfm ws init` writing the object), `build_mipmaps` with `-mip0`,
   `MipLevelFile::open` and `decode`, the staleness test, the writers' rules,
   `.gitignore` and `CACHEDIR.TAG`. Tests on 300 x 200 synthetic photographs
   in a temporary workspace whose `mipmap_options` sets a threshold of 64,
   with a tile of 32: the names for `IMG_0001.JPG` (`IMG_0001.JPG-mip1-q95.jpg`,
   case kept), `a.b.png` and a name with no extension, the names of `x.jpg`
   beside `x.png` differing; a mip level file whose
   record names another path is stale; one whose `source` differs from the
   photograph's path only in case and whose length, modification time and
   hash match the original is current, hashed once and not made again (one
   original under two spellings); one whose `source` differs only in case but
   whose original has another length or hash is stale and made again (two
   originals sharing a cache file); and one whose `source` differs only in
   case, whose length and modification time match the original but whose
   hash does not, is stale and made again (two originals of equal length and
   time); all built by writing the record directly, so the tests run on any
   file system; a workspace without the object and a photograph with no
   workspace are in core and write nothing; with the small threshold, a
   65535 x 100 photograph is out of core and a 65536 x 100 one refused with
   `GetError::TooLarge` naming it and its size, the latter also in a
   workspace without the object and with no workspace, each a synthetic PNG
   of which only the header is read (`read_layout` and `get` decide from it),
   never decoded at that size; two builds give the same bytes; a
   region-readable JPEG gets no `-mip0` and a PNG, a progressive JPEG, a
   JPEG without `DRI` and one with an interval wider than 1024 px do; each
   staleness case (length, time with equal hash, time with another hash,
   another `source`, a record whose quality or level disagrees with the
   name, another region answer, cut short, no record) and that a stale mipmap
   is made again; a change of `jpeg_quality` makes files under new names and
   leaves the old ones unread and current, and changing it back reads the
   old ones without making anything; a record of another threshold
   is still current; each invalid `mipmap_options` (a missing key, an unknown
   key, a float, a string, a quality of 0 or 101, a threshold of 0) is an
   error naming the key; a read-only cache directory gives the in-memory
   equivalent with the same pixels; two threads and two processes building at
   once. Outside the tests, the peak memory of one build of a 200-megapixel
   photograph is measured once and the estimate in § "How the levels are made"
   replaced by it, and the region test is run on all 17 of the user's
   photographs.
6. **Tile and region decodes**: the restart index, `MipLevelFile::decode_region`
   and `OriginalRegions`. Tests: every tile of several synthetic levels with
   odd sizes and a restart interval of 2 and 16 MCUs, including the short last
   interval, equals the same pixels of `decode`; a grey mip level file; a damaged
   interval is `Stale`; for JPEGs written by `jpeg-encoder` with restart
   intervals at 4:2:0, 4:2:2, 4:4:0, 4:4:4 and grey, of odd sizes, every
   region (including edges and corners) equals the same pixels of `read_rgb`;
   each rule of the region test refuses a file made to break it alone.
7. **View-region pyramids** ([view-region-pyramid.md](view-region-pyramid.md)
   steps 1 and 2): regions in `ImageU8Pyramid` and every sampler, the
   tile-addressed sampling path for renders the kernel call's region pool cannot hold, `required_regions`,
   `PhotographSource`, `clipped_share` over tiles, the grey reads by region,
   and the assembly from tiles, exercised on in-core photographs cut into
   tiles in memory. No reader changes yet. Tests: the companion draft's
   equality tests.
8. **The out-of-core tier, off.** The tiled entries, the tile budget and its
   eviction, the layout states, the slot as a mutex and condition variable
   with cancellation between a build's stages, the cache's build budget and
   its count of builds in progress (`PhotographCache::reserve_builds`), `get`
   and `get_many` answering sources and
   making missing mipmaps in the slot (the out-of-core misses of a `get_many` in
   groups sized by the budget), `get_level`, `TileError::Changed`, Image
   Detail and the 3D background on tiled photographs, with at most one
   out-of-core request in flight across the viewer, and the in-core admission
   rule. Every other
   reader refuses a tiled source. Tests: with the small threshold, a `get` on
   a photograph with no mipmap makes it and returns its pixels; two
   threads asking for one path make one mipmap; a `get_many` over 20 out-of-core
   photographs holds no pyramid (the cache's stats show only tiles, within a
   small tile budget) and never has more builds in progress than the budget
   allows, with a budget set small for the test and checked through the
   count; several `get_many` callers on one cache started at the same moment
   on separate threads never reserve more than the budget plus one mipmap each,
   two caches in one process keep separate counts, and their
   builds never add to the count beyond what was reserved; a burst of panel
   requests for twenty photographs without mipmaps starts their builds one at a
   time; a caller cancelled mid-build stops within one stage and leaves the
   slot empty with its share given back, a waiter on that slot then reserves
   and makes the mipmap, and a waiter that is
   cancelled stops waiting within 50 ms; a build reports its phases through
   the caller's `Progress`; no GUI-thread reader requests a tile while its
   photograph is `Building`, and none calls `get` for an out-of-core
   photograph on the GUI thread, with a pyramid budget of 0 too; a file
   replaced by one with another record during a read fails with `Changed`, and
   one made stale during a kernel fails it with the named error; `PhotographReadError` and the `render_source_*` functions, with
   the step-8 rows of § "Errors from reading pixels"; a reader not yet moved
   refuses a tiled source with `NotYetTiled`.
9. **Track View and the Track Table** on view-region pyramids, peeking tiles
   (companion step 3), with the step-9 signature changes.
10. **The bench's jobs and the reference picks** (companion step 4), with the
    step-10 signature changes, and one region pool made by each bench step
    and each reference pick for its kernel call.
11. **Bulk kernels**: the display patch bitmaps, the conversion to embedded
    patches, the web export, *Refine all normals*, *Refit all points* and the
    cluster-patches build (companion step 5), with the step-11 signature
    changes.
12. **The nearby-track grey reads by region** (companion step 6), with the
    step-12 signature changes.
13. **Makers**: `sfm sift` and `build_sift_file`, the viewer's *Build
    mipmaps* quiet task, `sfm ws mipmaps` with `--prune`
    and `--check` (its `--enable` waits for step 15, since it writes the
    object), and the thumbnail from the coarsest mip level file.
    Tests: `sfm sift` on a small-threshold synthetic workspace writes the
    files; the quiet task makes the selected image's mipmap in its first group
    and gives way to an asked-for task, leaving no slot marked as being made;
    `--prune` deletes an orphan and
    keeps a current mipmap, and deletes every mip level file in a workspace without the
    object; `--check` writes nothing.
14. **Python and the readers that must agree with the viewer**: the
    `sfmtool.photographs` submodule with the `PhotographCache` class (its
    layout, level, tile and region reads, `build_mipmaps`, `stats`) and core's
    `read_layout` and `read_region` behind it; `ImagePyramidSet.read` and
    `write_web_export` with `cache=`; `sfm ws mipmaps` and the Python SIFT
    extractors on `PhotographCache.build_mipmaps`; and `embed-patches`, the
    strips, the `xform` steps, `sfm cluster-patches` and the `scripts/`
    harnesses on `ImagePyramidSet.read`, with its sources raising
    `PhotographReadError` through the bindings and `NotYetTiled` deleted.
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
15. **The tier on.** `sfm ws init` writes `mipmap_options` with its three
    values, `sfm ws mipmaps --enable` adds it to an existing workspace, the
    viewer's one-line notice for a workspace without it, and the workspace
    spec gains the object. Tests: a new workspace has the object with 4096,
    95 and 95; `--enable` writes the object and refuses a file that has one;
    the test data's workspaces, which have no object, still write nothing.

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

## Open questions

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
   time on each cache's non-blocking count, each caller still granted one when
   the budget is used up, with at most one panel request in flight (the
   budget plus three mipmaps in the viewer); not a blocking semaphore, which
   would make a rayon worker wait for another caller's permit, and not a
   count per `get_many`, which would let two tasks and a panel each take the
   budget. A cancelled maker abandons its build and a waiter takes it over,
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
10. **Budgets.** *Decided by the user:* ship the defaults (a tile budget of a
    sixteenth of memory, 512 MiB to 4 GiB, and 256 GPU tile textures), report
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
    (§ "The workspace setting").

## Problems found with the decisions

- **Mirroring the workspace covers only photographs in a workspace.** A
  photograph with no workspace above it is in core, as are all photographs of
  a workspace without `mipmap_options`; a workspace whose cache directory
  cannot be written reads its out-of-core photographs through the in-memory
  equivalent.
- **A workspace without `mipmap_options` keeps today's memory.** Every
  existing workspace, the user's included, reads its large photographs as
  whole pyramids until the object is added: 762 MiB for each 200-megapixel
  photograph a panel or kernel reads, 12.6 GiB for a kernel over 17 of them.
  The display no longer crashes (steps 2 and 3), but the memory and time the
  tier saves are only had by adding the object (`sfm ws mipmaps --enable`),
  which the viewer tells the user once per session (open question 13).
- **"Every later reader reads the files" needs a rule for when files cannot be
  written.** The in-memory equivalent is that rule; its pixels equal the files'
  only with the same build (open question 8). For a photograph that needs a
  `-mip0`, it holds up to 58 MB of encoded bytes in the tile budget.
- **A coarsest level of about one tile is not deep enough for the kernels.**
  They need levels to about one pixel. The levels below the coarsest mip level are
  computed in memory from it, which keeps every reader consistent because the
  input is the file.
- **Whole pyramids for the kernels do not scale.** Today `get_many` hands
  back every photograph's pyramid and the caller holds them all while its
  kernel runs, 12.6 GiB for 17 photographs of 200 MP. So the design gives
  kernels no whole pyramid of an out-of-core photograph, and reads it only
  through view-region pyramids, which it requires.
- **Region reads depend on the decoder's internals.** The region test admits a
  layout only because `jpeg-decoder` 0.3.2 gives equal pixels for it with the
  margins found; a decoder change can change which originals get a
  `-mip0` (open question 9).
- **A quality setting per workspace means files go stale when it changes.**
  Every mipmap at the old quality is made again as it is next read, a whole
  decode and encode per photograph; the setting is meant to be chosen once.
  A change of the threshold moves photographs between tiers, which changes
  their results; their files are made or left as orphans, not rewritten.
- **Not every reader reads level 0 from `-mip0`.** Decisions 2 and 3 say
  every reader reads the stored levels once they exist. The pyramid readers
  do; the readers that decode a whole original and build no pyramid do not:
  SIFT extraction, the image readings at save, and `read_image_rgb` with the
  commands built on it (`undistort`, `to-nerfstudio`, the flow loader, the
  drawing commands, `xform --add-thumbnails`) read the original, as today.
  They run before the files exist or store what they compute from the exact
  pixels, and making them read a lossy copy would change the features and
  readings every reconstruction is built from. For an original that is not
  region-readable, SIFT's level 0 and the kernels' therefore differ by 0.07 to
  0.09 grey levels on the user's JPEGs (open question 11).
- **Builds on rayon workers slow the panels.** The user has decided that a
  mipmap is made the way a photograph is decoded today, in its slot on the
  caller's thread, so a rayon worker that makes a mipmap or waits on one is
  blocked for 2 to 4 s at 200 MP, against 0.5 to 1.7 s for today's decode.
  The panels' tile decodes and layout requests share the pool and wait longer
  for a free worker while builds run. This is accepted; what keeps it bounded
  is the build budget (open question 6), and what keeps the GUI responsive is
  that no build or wait ever runs on the GUI thread.
- **A build holds a whole decode.** Without the budget, a `get_many` over
  many out-of-core photographs would hold one 1 GB decode per rayon worker,
  32 GB on the development machine; the groups sized by the build budget and
  reserved on the shared count when they are decided, and the one panel
  request in flight, bound it to the budget plus one mipmap per caller, three in
  the viewer.
- **The cache directory is reachable by image discovery.** A directory scan
  at the workspace root would read mip level files as images; step 5 makes every
  finder skip it before anything writes there.
