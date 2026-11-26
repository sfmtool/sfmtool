# Mipmaps

A photograph of 200 megapixels takes a second or more to decode and about a
gigabyte to hold, and a viewer or kernel that needs only a corner of it, or a
small copy of it, should not pay that each time. A photograph's **mipmap** is
its stored chain of reduced levels, each a JPEG file under the workspace's
**mipmap directory**, `.sfm-mipmaps/`, made once from one decode of the
original; when the original cannot be read a rectangle at a time, the mipmap
also holds a full-resolution copy that can. One stored level is a **mip
level**, numbered as the pyramid's levels are (level 0 full resolution, level
`l` reduced by `2^l`), and the file holding it is a **mip level file**. A
workspace turns this on with a `mipmap_options` object in its
`.sfm-workspace.json`, which also sets the size above which a photograph is
**out of core** and the JPEG qualities.

This spec describes the files: which photographs have them, their names and
contents, how they are made, when they are current, and how a rectangle of
one, or of a region-readable original, is decoded alone. The photograph cache
reads an out-of-core photograph whose mipmap is current from its files, and
its `get` makes the mipmap of one whose files are missing or stale
([photograph-cache.md § "Out-of-core photographs"](photograph-cache.md#out-of-core-photographs)).
`build_sift_files`, behind the viewer's *Build SIFT Files* and `pick-images`,
makes the mipmap of each out-of-core photograph it decodes, from that decode
([sift.md § "Building `.sift` files"](../features/sift.md#building-sift-files)).
`sfm sift --extract` makes the mipmaps of the out-of-core photographs it
extracted once their `.sift` files are written, through
`PhotographCache::build_mipmaps`
([sift-command.md § Mipmaps](../../cli/image-feature/sift-command.md#mipmaps)).
`sfm ws mipmaps` makes the mipmaps of the photographs under its paths the same
way, and with `--check` reports which are missing or stale through
`check_mipmaps`, and with `--prune` deletes a mipmap directory's orphan files
through `prune_mipmaps`
([ws-mipmaps-command.md](../../cli/workspace/ws-mipmaps-command.md)).
`sfm ws init` writes `mipmap_options` into every new workspace
([§ "The workspace setting"](#the-workspace-setting)). A workspace made
before it has no object until its owner adds one; `sfm ws mipmaps --enable`,
which adds it, is planned in step 17b of
[out-of-core-photograph-pyramid.md](../../drafts/out-of-core-photograph-pyramid.md),
which also has the measurements behind the choices below.

## Rust API

[mipmap.rs](../../../crates/sfmtool-core/src/camera/mipmap.rs) and its
submodules, exported from `sfmtool_core::camera::mipmap` and, for the main
names, `sfmtool_core::camera`; the workspace object's reader is
[workspace/mipmap_options.rs](../../../crates/sfmtool-core/src/workspace/mipmap_options.rs).

```rust
pub const MIPMAP_DIR_NAME: &str = ".sfm-mipmaps";
pub const DEFAULT_OUT_OF_CORE_LONG_SIDE: u32 = 1536; // what `sfm ws init` writes
pub const MAX_JPEG_SIDE: u32 = 65535;
pub const PHOTOGRAPH_TILE_SIZE: u32 = 512;
pub const MIPMAP_RESTART_INTERVAL_MCUS: u16 = 16;
pub const MAX_REGION_INTERVAL_WIDTH: u32 = 1024;

/// A workspace's `mipmap_options`; every field is required in the file.
pub struct MipmapOptions { pub out_of_core_long_side: u32, pub jpeg_quality: u8,
                           pub full_resolution_jpeg_quality: u8 }

/// `sfmtool_core::workspace`: the object in `<root>/.sfm-workspace.json`,
/// `None` when the file has none, an error naming file, key and value.
pub fn read_mipmap_options(root: &Path) -> Result<Option<MipmapOptions>, WorkspaceConfigError>;

/// `sfmtool_core::workspace`: the object a new workspace records (1536, 95
/// and 95), and an object as the file holds it, which `read_mipmap_options`
/// reads back.
pub const NEW_WORKSPACE_MIPMAP_OPTIONS: MipmapOptions;
pub fn mipmap_options_json(options: &MipmapOptions) -> serde_json::Value;

/// Level sizes, the tier, and `K`, the coarsest stored level. `None` for a
/// side over `MAX_JPEG_SIDE`.
impl PyramidLayout {
    pub fn for_size(width: u32, height: u32, options: Option<&MipmapOptions>, tile_size: u32) -> Option<Self>;
    pub fn levels(&self) -> usize;
    pub fn level_size(&self, level: usize) -> (u32, u32);
    pub fn stored_levels(&self) -> usize; // 0 in core
    pub fn is_out_of_core(&self) -> bool;
    pub fn tiles(&self, level: usize) -> (u32, u32);
}

/// The layout under the photograph's workspace's options, read from its
/// header, and its mipmap's location when out of core.
pub fn photograph_layout(photograph: &Path, tile_size: u32)
    -> Result<(PyramidLayout, Option<MipmapLocation>), MipmapError>;

/// The region test: what a region read needs, or the first rule failed.
pub fn region_layout(bytes: &[u8]) -> Result<RegionLayout, NotRegionReadable>;

/// The workspace root, relative path and directory of a photograph's files;
/// `None` outside every workspace.
pub fn mipmap_location(photograph: &Path) -> Option<MipmapLocation>;
impl MipmapLocation { pub fn level_path(&self, level: usize, quality: u8) -> PathBuf; /* … */ }

/// Make the mipmap from one decode, writing what can be written, telling
/// `on_level` each level as it is encoded, the coarsest first.
pub fn build_mipmaps(photograph: &Path, decoded: Option<&ImageU8>, options: &MipmapOptions,
                     tile_size: u32, progress: &Progress<'_>,
                     on_level: &mut dyn FnMut(&MipLevelFile)) -> Result<BuiltMipmap, MipmapError>;
pub struct BuiltMipmap { pub layout: PyramidLayout, pub region_readable: bool,
                         pub files: Vec<MipLevelFile>, pub not_written: Option<String> }

/// Whether the files on disk are current; the region answer they record.
pub fn check_mipmap(source: &MipmapSource, location: &MipmapLocation, layout: &PyramidLayout)
    -> Result<bool, MipmapError>;
/// The original's facts a record is checked against, hashed at most once.
impl MipmapSource { pub fn read(photograph: &Path, location: &MipmapLocation) -> Result<Self, MipmapError>;
                    pub fn hashes(&self) -> usize; /* … */ }

/// Whether the mipmaps of many photographs are current, writing nothing;
/// and the deletion of a mipmap directory's orphan files, or a list of them.
/// Both are `sfm ws mipmaps`', specified in ws-mipmaps-command.md.
pub fn check_mipmaps(photographs: &[&Path], tile_size: u32, progress: &Progress<'_>)
    -> Result<MipmapCheckReport, Cancelled>;
pub fn prune_mipmaps(paths: &[&Path], options: &PruneMipmapsOptions, progress: &Progress<'_>)
    -> Result<MipmapPruneReport, Cancelled>;

/// One file's bytes, record and restart index, read and checked once.
impl MipLevelFile {
    pub fn open(path: &Path, expected: &ExpectedMipLevel) -> Result<Self, MipmapError>;
    pub fn from_bytes(bytes: Arc<[u8]>, expected: &ExpectedMipLevel) -> Result<Self, MipmapError>;
    pub fn record(&self) -> &MipLevelRecord;
    pub fn decode(&self) -> Result<ImageU8, MipmapError>;
    /// Pixels `x0..x1` by `y0..y1`, equal to the same pixels of `decode`.
    pub fn decode_region(&self, x0: u32, y0: u32, x1: u32, y1: u32) -> Result<ImageU8, MipmapError>;
}

/// Region reads of a region-readable original, each equal to the same
/// pixels of `ImageU8::read_rgb` of the file.
impl OriginalRegions {
    pub fn from_bytes(bytes: Arc<[u8]>) -> Result<Self, NotRegionReadable>;
    pub fn width(&self) -> u32;
    pub fn height(&self) -> u32;
    pub fn decode_region(&self, x0: u32, y0: u32, x1: u32, y1: u32) -> Result<ImageU8, image::ImageError>;
}

pub enum MipmapError { Io { .. }, Stale { path, why }, Source { .. }, Workspace(WorkspaceConfigError),
                       TooLarge { path, width, height }, NotOutOfCore { path }, OutOfRange { path, why },
                       Encode(String), Cancelled }
```

Why it is shaped this way:

- **Everything is found from the photograph's path.** The workspace root, the
  options and the file names follow from the path, so the photograph cache
  keeps taking paths.
- **A build returns the files it encoded.** The caller decodes those bytes
  rather than reading back the files just written, and when the files could
  not be written the same bytes are the in-memory equivalent (below).
- **A build tells each level as it is encoded.** The coarsest level is
  encoded first, about one decode's time after the build starts and before
  any file is written, so a caller that shows the photograph while the rest
  is made has it then: the photograph cache publishes it for the panels to
  draw ([photograph-cache.md § "Making a mipmap"](photograph-cache.md#making-a-mipmap)).
  A callback rather than a second function, so there is one build.
- **The original's facts are a value of their own.** `MipmapSource` hashes the
  original at most once however many files are checked against it, so a
  caller that keeps one per original and spelling of its path hashes once.
- **No builder object.** `build_mipmaps` is a function: the photograph cache
  is to call it in a path's slot, so only one build per path and cache runs.
- **A file is read and indexed once, then decoded a rectangle at a time.**
  `MipLevelFile` and `OriginalRegions` hold the bytes and the restart index,
  so each tile costs only the intervals it needs; the photograph cache keeps
  one per file as an entry of its own, counted against its budget.

Example: is a photograph's mipmap current, and if not, make it.

```rust
let (layout, location) = photograph_layout(&path, PHOTOGRAPH_TILE_SIZE)?;
if let (Some(location), Some(options)) = (location, layout.options()) {
    let source = MipmapSource::read(&path, &location)?;
    if check_mipmap(&source, &location, &layout).is_err() {
        build_mipmaps(&path, None, options, PHOTOGRAPH_TILE_SIZE, &Progress::none(), &mut |_| {})?;
    }
}
```

Example: the top-left 512 px tile of level 1, with its one-pixel border, from
a current mipmap.

```rust
let expected = ExpectedMipLevel::new(&layout, location.relative(), 1).expect("out of core");
let file = MipLevelFile::open(&location.level_path(1, options.jpeg_quality), &expected)?;
let (width, height) = layout.level_size(1);
let tile = file.decode_region(0, 0, 513.min(width), 513.min(height))?;
```

## Which photographs are out of core

A photograph is out of core when its workspace's file has `mipmap_options` and
its width or height, read from its header with `image_dimensions`, is greater
than the object's `out_of_core_long_side`. Any other photograph is in core, and
always in a workspace without the object or with no workspace at all: it has no
mip level files, and nothing is written for it.

**A photograph with a side of 65536 px or more is refused** by the photograph
cache in every workspace and with none, decided from the header before any
decode: `PhotographCache::get` answers `GetError::TooLarge { path, width,
height }`, and `photograph_layout` `MipmapError::TooLarge`. Every file the
mipmap may write, `-mip0` included, must fit a JPEG frame, whose sides are at
most 65535, and one rule for every workspace keeps a photograph's readability
the same whatever the workspace sets. Only a non-JPEG original can be that
large. `read_image_rgb` and the readers built on it keep their own 4 GiB
decode limit.

## Levels by size

Level `l` is `floor(width / 2^l)` by `floor(height / 2^l)`, as repeated
`downsample_2x` makes it, and there are `ImageU8Pyramid::full_levels(width,
height)` levels. An out-of-core photograph's mipmap stores levels 1 to `K`,
where `K` is the first level whose longer side is at most one photograph tile
(512 px), or the last level whose sides are both at least 1 px if that comes
first; the levels below `K` are to be box-filtered from it in memory. It also
stores level 0, `-mip0`, exactly when its original is not region-readable.

| Photograph | Levels | `K` | Coarsest mip level |
|------------|--------|-----|--------------------|
| 16320 x 12240 | 14 | 5 | 510 x 382 |
| 6000 x 4000 | 12 | 4 | 375 x 250 |
| 4097 x 2304 | 12 | 3 | 512 x 288 |
| 30000 x 1000 | 10 | 6 | 468 x 15 |

## Which originals can be read by region

`region_layout` reads the original's bytes and nothing else, so every reader
decides the same way for the same file. An original is **region-readable**
when all of these hold, checked in order; the first that fails is the
`NotRegionReadable` variant:

1. It starts with the JPEG start-of-image marker (`NotJpeg`): every PNG, TIFF
   and other format fails, whatever its name.
2. Its frame is `SOF0` or `SOF1` with 8-bit samples and sides that are not 0
   (`Frame`): progressive, lossless, arithmetic-coded and 12-bit frames fail.
3. It has one component, or three that `jpeg-decoder` 0.3.2 reads as YCbCr
   (`Colour`): by the component identifiers first (1, 2, 3 is YCbCr; `R`,
   `G`, `B` and the two big-gamut sets are not), then a JFIF segment (YCbCr),
   then an Adobe segment's transform (1 is YCbCr; 0 is RGB and 2 YCCK), and
   YCbCr when none decides, as its `determine_color_transform` does.
4. The chroma components are sampled 1 x 1 and luma `H x V` with each 1 or 2
   (`Sampling`).
5. One scan holds every component in frame order, with spectral selection 0
   to 63 and no successive approximation, and no second scan follows
   (`Scan`).
6. A `DRI` segment before the scan sets an interval `R > 0` that divides the
   MCUs per row (`ceil(width / 8H)` with three components, `ceil(width / 8)`
   with one, whose scan is not interleaved), so each interval lies in one MCU
   row, and an interval is at most 1024 px wide (`RestartInterval`). Wider
   intervals would make every tile decode a strip many tiles wide.
7. Indexing the scan finds exactly `rows · (MCUs per row / R)` non-empty
   intervals separated by `RST0` to `RST7` in sequence modulo 8 and then
   end-of-image (`Intervals`). Bytes after end-of-image are ignored.

The test is part of the definition of a photograph's pixels: changing it moves
originals between being read directly and being read from `-mip0`, so it
changes only with a test showing why.

## Names

```
<root>/.sfm-mipmaps/<parent of relative path>/<file name>-mip<l>-q<quality>.jpg
```

The root is the closest ancestor holding `.sfm-workspace.json`
(`find_workspace_for_path`, which does not resolve links), and the relative
path is the photograph's path below it. `<file name>` is the original's whole
file name in its own case, extension included, as `.sift` file names take it
(`IMG_0001.JPG-mip1-q95.jpg`, `a.b.png-mip2-q95.jpg`, `README-mip1-q95.jpg`),
so `x.jpg` and `x.png` beside each other have different files. The number
after `mip` is the level; `quality` is the file's JPEG quality in decimal,
`full_resolution_jpeg_quality` for `-mip0` and `jpeg_quality` for the rest.
A file at another quality is another file, so changing a quality makes new
names and leaves the old files current for a change back. The threshold, the
tile size and the restart interval are not in the name: none changes a file's
content.

## What is written in one

A baseline JPEG any program can open, written by `jpeg-encoder`: 4:4:4 colour,
or one component when every pixel of level 0 has R = G = B; standard Huffman
tables (optimized ones put each component in a scan of its own); a restart
interval of 16 MCUs; the JFIF header the encoder writes; and one APP15 segment
holding the **record**, the ASCII bytes `sfmtool-mipmap`, a zero byte, and a
JSON object:

| Field | Holds |
|-------|-------|
| `version` | 1 |
| `source` | the original's relative path, with `/` |
| `source_size`, `source_modified_ns`, `source_xxh128` | the original's length, modification time (ns since the Unix epoch, `null` where there is none) and XXH128 as 32 hex digits, as a `.sift` file records an image's |
| `source_width`, `source_height` | the original's size |
| `level`, `width`, `height` | the level the file holds and its size |
| `out_of_core_long_side` | the threshold it was made under, for diagnosis only |
| `region_readable_original` | the region test's answer |
| `stored_levels` | `K` |
| `jpeg_quality` | the quality it is encoded at, equal to the one in its name |
| `components`, `restart_interval_mcus` | 1 or 3, and 16 |
| `encoder`, `sfmtool_version` | what wrote it |

The record is in each file rather than a manifest beside them, so a file
cannot be separated from the facts about it and two writers never edit one
shared file. A mip level file is not a file format in the sense of the
[I/O-shape standard](../../STANDARDS.md#every-file-format-has-the-same-io-shape):
nothing outside sfmtool reads the record, so it has no format crate and no
`sfmtool.fileio` functions; this section is the one place that defines it.

## The workspace setting

```json
"mipmap_options": {
  "out_of_core_long_side": 1536,
  "jpeg_quality": 95,
  "full_resolution_jpeg_quality": 95
}
```

When the object is present it holds exactly these three keys, each a JSON
integer: the threshold at least 1, the qualities 1 to 100. A missing key, an
unknown key, a value that is not a JSON integer (`95.0`, `"95"`, `true`) or
one out of range is a `WorkspaceConfigError` from `read_mipmap_options` that
names the file, the key and the value. There are no defaults inside the
object, so what a workspace's mipmaps were made with is always in its file.
Without the object there are no mipmaps. The photograph cache reads the object
once per workspace root and again when the file changes, and refuses every
photograph of a workspace whose object is invalid until the file is fixed
([photograph-cache.md § "A changed file and remembered failures"](photograph-cache.md#a-changed-file-and-remembered-failures)).

`sfm ws init` writes the object above, `NEW_WORKSPACE_MIPMAP_OPTIONS`, into
every new workspace, after `feature_prefix_dir`, for each feature tool
([ws-init-command.md](../../cli/workspace/ws-init-command.md#persisted-settings)).
The threshold of 1536 px keeps memory bounded for a job that reads many
photographs of a few megapixels, such as those of a 24-megapixel camera or a
phone: an in-core photograph is decoded whole, all its pixels at once, when a
tile of it is needed and the cache holds none, and decoded whole again once
its tiles are evicted, while an out-of-core one decodes only the tiles a read
needs, about 3 ms each, at the cost of one encode of its levels. The
threshold is separate from `double_image_below_long_side`, which a new
workspace sets to 4096: a photograph whose longer side is 1537 to 4095 px is
out of core and is doubled for SIFT. SIFT decodes the original whole in either
tier, and the mipmap is made from the photograph, not the doubled image: the
Rust `.sift` build makes it from its decode before the extraction
([sift.md § "Building `.sift` files"](../features/sift.md#building-sift-files)),
and `sfm sift` decodes the original again once the files are written. Quality
95 re-encodes a JPEG original's full-resolution copy closest to the original
among the qualities measured (0.07 to 0.09 grey levels), and no quality
depends on the original's format; a workspace of PNG or TIFF photographs can
set `full_resolution_jpeg_quality` higher itself. Each workspace keeps the
values in its own file, so a later change of these defaults does not change
which pixels an existing workspace's kernels read.

## How a mipmap is made

`build_mipmaps` first reads the photograph's header (`image_dimensions`), so a
photograph that is in core under the options, or has a side over 65535 px, is
refused without being read whole or decoded. It then reads the original once,
decodes it with the decoder every reader uses (`ImageU8::decode_rgb` of the
bytes, equal to `read_rgb` of the file) unless the caller passes the decode,
refuses a decode that is not the header's size, runs the region test on the
same bytes, and makes levels 1 to `K` by `downsample_2x`, each from the exact
level above it in memory. Each level is encoded on its own, coarsest first, and
`-mip0` from the decode last, since it is the largest. Every file therefore
carries one JPEG loss, any level can be made again alone to the same bytes,
and the reduced levels are the same whether or not the original is
region-readable. The same original and build give the same bytes.
`on_level` is told each level's file as soon as it is encoded, or read
when it is current, before it is written: the coarsest first, then each
finer level and `-mip0` last.

A level whose file is current already (§ "When a mip level file is
current", with the build's number of stored levels and region answer) is
read from that file rather than encoded and written again, so a build that
follows one that stopped makes only the levels not written. The coarsest
level's file is written last, after `-mip0`: a reader that checks only the
coarsest file, as the photograph cache does when it first needs a
photograph, never finds it current beside finer files a stopped build did
not write. A file at a level's name whose `source` differs from the
photograph's relative path only in letter case and that the hash finds is
another original's is made over, with one warning through the caller's
`Progress` naming both paths.

It reports the phases `decode`, `reduce` and `encode` through the caller's
`Progress`, counting levels, and polls for cancellation between levels and
before the coarsest file is written; a cancelled build writes no further file
and leaves none partial.

**Writes.** Each file is written to a temporary file in its directory,
`.<file name>.<process id>-<count>.tmp` with the one process-wide count
`ImageU8::write` also uses, and renamed into place. Unlike `ImageU8::write`, a
build whose rename fails does not write the target directly: a target that is
now current for the original counts as written (another writer made it), and
otherwise the build writes nothing further and keeps the encoded bytes, its
`not_written` saying why. The temporary file is removed either way. Two
builders of one photograph, in one process or in two, write the same bytes and
the last rename wins; a reader holding a file open does not stop another
process renaming over it.

**The in-memory equivalent.** When the files cannot be written (a read-only or
blocked mipmap directory, a full disk), the build's `files` are the pixels the
files would have held, at the workspace's qualities, so a reader can decode
them rather than reading the exact levels for one photograph and the JPEG ones
for another. The same original and build give the same bytes, so a reader that
has dropped them makes them again rather than keep them for its life
([photograph-cache.md § "Making a mipmap"](photograph-cache.md#making-a-mipmap)).

## When a mip level file is current

`check_mipmap` checks every stored level from the coarsest, then `-mip0` when
the records say the original is not region-readable. A file is current when
all of these hold, in order, and stale naming the first that fails:

1. It reads as a JPEG with a record of a known version, the record's `level`
   and `jpeg_quality` equal those in its name, its frame is the record's
   `width` x `height` at 4:4:4 or grey with the record's restart interval, and
   its scan indexes whole (a file cut short fails here).
2. Its `source` is the photograph's relative path, or differs from it only in
   letter case (both `str::to_lowercase`d compare equal), and its size is the
   level's size for the original's.
3. The original's header size is `source_width` x `source_height`.
4. The original's length is `source_size` and its modification time
   `source_modified_ns`. When the length matches and the time does not, or
   when `source` matched only after lower-casing, the original is hashed and
   compared with `source_xxh128`. A different length is stale without a hash.
   So one original reached by two spellings shares its mipmap, and two
   originals whose names differ only in case, sharing one file on a
   case-insensitive file system, never read each other's pixels.
5. Its `region_readable_original` and `stored_levels` equal the other files'.
6. When the records say the original is not region-readable, its `-mip0` is
   present and current.

The quality is not compared with the workspace's setting: only names of the
current setting are opened. Nor is the threshold: a mipmap whose photograph is
still out of core stays current whatever threshold its record names. A file
that is missing, damaged or stale is never an error to the caller beyond
`Stale`, and never a panic; the remedy is a new build.

## Reading a rectangle

A tile of an out-of-core photograph's level is decoded from the file that
holds the level without decoding the rest of it: a tile of a mip level file by
`MipLevelFile::decode_region`, and a region of level 0 of a region-readable
original by `OriginalRegions::decode_region`. Both rely on the restart markers
in the file: each restart interval starts with the DC predictions reset, so it
decodes the same wherever it is placed.

**The restart index** is made once per file, when it is opened (`open`,
`from_bytes`): the walk of the entropy-coded data that test 1 of when a file is
current, or rule 7 of the region test, makes anyway, kept as the byte range
of each interval, with the segments a decode reads (every segment up to and
including the scan header but the APPn and COM segments). A rectangle is then
decoded by assembling a small JPEG: those segments with the frame's size
changed, the intervals the rectangle needs with restart markers between them
numbered again from `RST0` modulo 8, and end-of-image. It is decoded with
`jpeg-decoder` 0.3.2. Leaving out the APPn segments, among them the JFIF and
Adobe segments the decoder reads the colour transform from, changes nothing:
a mip level file is decoded with `ColorTransform::None` and an original with
`ColorTransform::YCbCr`, the reading rule 3 requires.

### A tile of a mip level file

With `M` MCUs per row and `N` per interval, the MCU at row `r`, column `c` is
in interval `(r·M + c) div N`, so an interval may run from the end of one MCU
row into the next. The intervals a rectangle needs are the sorted union, over
the MCU rows it spans, of those holding the MCUs that cover its columns. Each
becomes one MCU row of a small frame `8·N` wide, wherever it lies in the
level, and each of its MCUs is copied from there to its place, cropped to the
rectangle. The level's last interval holds fewer than `N` MCUs when the
level's MCU count is not a multiple of `N`; it is decoded in a frame of its
own, as wide as it is. A frame holds at most 8191 intervals, so its height
fits a JPEG frame, and a larger rectangle takes several.

The small frame is decoded with `ColorTransform::None`, which returns the
samples as coded, and converted to RGB with the same fixed-point formula as
`decode` (§ "Implementation notes"), so every pixel of a tile equals the same
pixel of `decode`. Two choices make that hold:

- **The conversion outside the decoder.** `jpeg-decoder`'s SSSE3 conversion
  (`color_convert_line_ycbcr` in its `arch/ssse3.rs`) converts all but the
  last 8 to 15 pixels of each line and leaves those to its scalar formula,
  which rounds some pixels one grey level differently. The small frame's lines
  end at other places than the level's, so converting inside the decoder
  would make a pixel's value depend on where its line ends.
- **4:4:4.** A mip level file has no chroma subsampling, so an 8 x 8 block's
  pixels depend only on its own coefficients; with subsampled chroma the
  decoder's upsampling reads the neighbouring MCUs', which in the small frame
  are other intervals.

A damaged interval that no longer decodes, or decodes to the wrong number of
samples, makes the tile `MipmapError::Stale`; a tile that does not need it
still reads. An empty rectangle or one not inside the level is
`MipmapError::OutOfRange`.

### A region of a region-readable original

An original is usually subsampled, so its region decode keeps the original's
layout in two dimensions instead. The small frame's columns are the
rectangle's MCU columns widened to interval boundaries, its rows are the
rectangle's MCU rows, both with a margin, and its intervals are copied in
raster order; rule 6 puts each interval in one MCU row. It is decoded to RGB by
the same `jpeg-decoder` path `ImageU8::read_rgb` takes, and cropped. The
margins make every cropped pixel equal the whole decode's:

- one MCU above, below and to the left, for the neighbours chroma upsampling
  reads (`jpeg-decoder` upsamples 4:2:0, 4:2:2 and 4:4:0 with the triangle
  filter libjpeg calls fancy upsampling, `UpsamplerH2V2` and its kin in its
  `upsampler.rs`);
- one MCU on the right, which also keeps the cropped pixels out of the part of
  each line the decoder converts with its scalar formula: in a frame a whole
  number of MCUs wide that is the last 8 px, inside the margin, and such a
  frame ends at least one MCU before the image's last, so in the whole decode,
  whose scalar part is the last 8 to 15 px of the image, the cropped pixels
  are converted by the SIMD formula too.

Where the frame reaches the image's right or bottom edge, its width or height
is the distance to that edge rather than a whole number of MCUs, so the edge
is padded and upsampled as in the whole decode. Since the frame starts on a
multiple of 8 px, the scalar part of each line then falls on the same pixels
in both decodes. A grey original needs none of this, and takes the same path.

**Strips no wider than 128 px.** `jpeg-decoder` 0.3.2 decodes a frame more
than 128 px wide on one new OS thread per colour component, which the calling
thread waits for (`select_worker` in its `decoder.rs`). A kernel keeps every
core busy with its own threads, so those threads wait their turn to run: a
514 px tile of a 16320 x 12240 original that decodes in about 3 ms on an idle
machine took about 200 ms on average inside `sfm xform --refine-normals`. A
region is therefore decoded as several frames, strips side by side, each at
most 128 px wide (`DECODED_ON_THE_CALLING_THREAD`), which the decoder decodes
on the calling thread. Each strip is a frame as above: it starts on an
interval boundary, holds one MCU of margin on each side of the columns it
gives unless it reaches the image's edge there, and stops at the one frame's
right edge, so each of its pixels equals the whole decode's for the same
reasons. Strips of an original whose interval is 64 px (4:2:0 with `R = 4`)
give 64 px each from a 128 px frame, so a tile decodes about twice its pixels;
one strip costs more than a share of the one wide frame, about 13 ms for a
514 px tile on one idle thread, but a kernel's threads no longer wait on
threads that cannot run. When the interval is too wide for 128 px frames to
cover the columns with their margins (more than 64 px, with MCUs 8 or 16 px
wide; rule 6 allows up to 1024 px), the region is one frame as wide as it
needs.

`OriginalRegions::from_bytes` runs the region test, so it is made only for a
region-readable original and its `Err` names the rule failed. A rectangle that
is empty or not inside the image is an `ImageError::Parameter`, and intervals
that do not decode an `ImageError::Decoding`.

This equality rests on how `jpeg-decoder` 0.3.2 upsamples chroma and where its
SIMD conversion stops. A decoder change that breaks the equality test has to
change the region test to send the affected layouts to `-mip0`, as a
deliberate change of results, rather than let region reads drift from whole
decodes.

## The mipmap directory

- A build that writes in `.sfm-mipmaps/` puts a `CACHEDIR.TAG` with the
  standard signature line there, which backup tools honour by skipping it.
- sfmtool writes no `.gitignore` and edits none; a user who keeps a workspace
  in git adds `.sfm-mipmaps/` to its own `.gitignore`.
- Every search for a workspace's images skips any directory named
  `.sfm-mipmaps`: `expand_paths`, the rig-config converter's sensor search,
  the `.camrig` pattern matcher and the named-image-list walk.
- No reader deletes a file. `sfm ws mipmaps --prune` deletes the orphan
  files, those no photograph would read, and nothing else in it
  ([ws-mipmaps-command.md § Orphans](../../cli/workspace/ws-mipmaps-command.md#orphans));
  deleting the directory by hand is always safe.

## Implementation notes

- **Decoding a whole level** uses `jpeg-decoder` with
  `ColorTransform::None`, which returns each line's Y samples, then its Cb,
  then its Cr, and converts them to RGB with the fixed-point BT.601 formula of
  its scalar path (`ycbcr_to_rgb` in its `decoder.rs`, from libjpeg-turbo's
  `jdcolext.c`). A tile decode converts with the same formula
  (§ "Reading a rectangle"). Converting outside the decoder costs time: a
  whole decode of a 4080 x 3060 level took 0.58 to 0.64 s, against 0.16 to
  0.45 s with the decoder's own SIMD conversion. The photograph cache
  decodes only a mipmap's coarsest level whole.
- **The marker walk** finds markers as `jpeg-decoder` does (bytes up to the
  next `FF` skipped, fill bytes skipped, `FF 00` not a marker), so the region
  test and the reader agree on which segments a file has.

## Testing

[mipmap/tests.rs](../../../crates/sfmtool-core/src/camera/mipmap/tests.rs)
runs on 300 x 200 synthetic photographs in a temporary workspace with a
threshold of 64 and tiles of 32: the files and `CACHEDIR.TAG` of a first build
and no `.gitignore`; `-mip0` for a PNG, a progressive JPEG, one without `DRI`
and one with an interval wider than 1024 px, and none for a region-readable
JPEG; two builds giving the same bytes; each staleness case, made by writing
the record directly so it runs on any file system, including the three
case-only `source` cases, and the remake; a quality change and back; a record
of another threshold; in-core photographs writing nothing; a build over
a file of an original whose path differs only in case warning once, naming
both; the 65535 / 65536
header-only refusal in every workspace kind; the in-memory equivalent behind a
blocked mipmap directory; a cancelled build; a grey photograph; each
level told as it is encoded, the coarsest first and before its file is
written, and a current level told as it is read; a build of a
header-only PNG answering in core or too large without decoding it; and two
threads and two processes building at once while a file is held open.
[region/tests.rs](../../../crates/sfmtool-core/src/camera/mipmap/region/tests.rs)
has the region test's cases, with a file made to break each rule alone,
[tile/tests.rs](../../../crates/sfmtool-core/src/camera/mipmap/tile/tests.rs)
the rectangle decodes: every 32 px tile, corner pixel and small rectangle of
synthetic levels of odd sizes, in colour and grey, with intervals of 2 and 16
MCUs that cross MCU rows and leave a short last interval, equal to the same
pixels of `decode`; `decode` within one grey level of the decoder's own RGB;
a damaged interval stale for the tiles that need it only; every region of
synthetic originals of odd sizes at 4:2:0, 4:2:2, 4:4:0, 4:4:4 and grey equal
to the same pixels of `read_rgb`; a 514 px tile, inside the image and at its
right edge, of originals with intervals of 48 and 64 px (4:2:0), 64 px
(4:4:4) and 8 px (grey) decoded in several frames, none wider than 128 px and
together at most about twice the tile's columns, equal to `read_rgb`, and a
region of an original with 112 px intervals decoded in one frame; out-of-range
rectangles refused; and every
cut and many single-byte changes of both kinds of file an error, never a
panic. [layout.rs](../../../crates/sfmtool-core/src/camera/mipmap/layout.rs) the
levels by size, and
[mipmap_options.rs](../../../crates/sfmtool-core/src/workspace/mipmap_options.rs)
each invalid `mipmap_options`.
