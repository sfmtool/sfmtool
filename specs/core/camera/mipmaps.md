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
contents, how they are made, and when they are current. No reader of
photograph pixels reads them yet, and no command writes `mipmap_options`; the
photograph cache, the viewer, the kernels and `sfm` move to them in the steps
of [out-of-core-photograph-pyramid.md](../../drafts/out-of-core-photograph-pyramid.md),
which also has the measurements behind the choices below.

## Rust API

[mipmap.rs](../../../crates/sfmtool-core/src/camera/mipmap.rs) and its
submodules, exported from `sfmtool_core::camera::mipmap` and, for the main
names, `sfmtool_core::camera`; the workspace object's reader is
[workspace/mipmap_options.rs](../../../crates/sfmtool-core/src/workspace/mipmap_options.rs).

```rust
pub const MIPMAP_DIR_NAME: &str = ".sfm-mipmaps";
pub const DEFAULT_OUT_OF_CORE_LONG_SIDE: u32 = 4096; // what `sfm ws init` is to write
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

/// Make the mipmap from one decode, writing what can be written.
pub fn build_mipmaps(photograph: &Path, decoded: Option<&ImageU8>, options: &MipmapOptions,
                     tile_size: u32, progress: &Progress<'_>) -> Result<BuiltMipmap, MipmapError>;
pub struct BuiltMipmap { pub layout: PyramidLayout, pub region_readable: bool,
                         pub files: Vec<MipLevelFile>, pub not_written: Option<String> }

/// Whether the files on disk are current; the region answer they record.
pub fn check_mipmap(source: &MipmapSource, location: &MipmapLocation, layout: &PyramidLayout)
    -> Result<bool, MipmapError>;
/// The original's facts a record is checked against, hashed at most once.
impl MipmapSource { pub fn read(photograph: &Path, location: &MipmapLocation) -> Result<Self, MipmapError>;
                    pub fn hashes(&self) -> usize; /* … */ }

/// One file's bytes and record, read and checked once.
impl MipLevelFile {
    pub fn open(path: &Path, expected: &ExpectedMipLevel) -> Result<Self, MipmapError>;
    pub fn from_bytes(bytes: Arc<[u8]>, expected: &ExpectedMipLevel) -> Result<Self, MipmapError>;
    pub fn record(&self) -> &MipLevelRecord;
    pub fn decode(&self) -> Result<ImageU8, MipmapError>;
}

pub enum MipmapError { Io { .. }, Stale { path, why }, Source { .. }, Workspace(WorkspaceConfigError),
                       TooLarge { path, width, height }, NotOutOfCore { path }, Encode(String), Cancelled }
```

Why it is shaped this way:

- **Everything is found from the photograph's path.** The workspace root, the
  options and the file names follow from the path, so the photograph cache
  keeps taking paths.
- **A build returns the files it encoded.** The caller decodes those bytes
  rather than reading back the files just written, and when the files could
  not be written the same bytes are the in-memory equivalent (below).
- **The original's facts are a value of their own.** `MipmapSource` hashes the
  original at most once however many files are checked against it, so a
  caller that keeps one per original and spelling of its path hashes once.
- **No builder object.** `build_mipmaps` is a function: the photograph cache
  is to call it in a path's slot, so only one build per path and cache runs.

Example: is a photograph's mipmap current, and if not, make it.

```rust
let (layout, location) = photograph_layout(&path, PHOTOGRAPH_TILE_SIZE)?;
if let (Some(location), Some(options)) = (location, layout.options()) {
    let source = MipmapSource::read(&path, &location)?;
    if check_mipmap(&source, &location, &layout).is_err() {
        build_mipmaps(&path, None, options, PHOTOGRAPH_TILE_SIZE, &Progress::none())?;
    }
}
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
  "out_of_core_long_side": 4096,
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
Without the object there are no mipmaps.

## How a mipmap is made

`build_mipmaps` reads the original once, decodes it with the decoder every
reader uses (`ImageU8::decode_rgb` of the bytes, equal to `read_rgb` of the
file) unless the caller passes the decode, runs the region test on the same
bytes, and makes levels 1 to `K` by `downsample_2x`, each from the exact level
above it in memory. Each level is encoded on its own, coarsest first, and
`-mip0` from the decode last, since it is the largest. Every file therefore
carries one JPEG loss, any level can be made again alone to the same bytes,
and the reduced levels are the same whether or not the original is
region-readable. The same original and build give the same bytes.

It reports the phases `decode`, `reduce` and `encode` through the caller's
`Progress`, counting levels, and polls for cancellation between levels; a
cancelled build writes no further file and leaves none partial.

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
for another.

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

## The mipmap directory

- A build that writes in `.sfm-mipmaps/` puts a `CACHEDIR.TAG` with the
  standard signature line there, which backup tools honour by skipping it.
- sfmtool writes no `.gitignore` and edits none; a user who keeps a workspace
  in git adds `.sfm-mipmaps/` to its own `.gitignore`.
- Every search for a workspace's images skips any directory named
  `.sfm-mipmaps`: `expand_paths`, the rig-config converter's sensor search,
  the `.camrig` pattern matcher and the named-image-list walk.
- No reader deletes a file; deleting the directory by hand is always safe.

## Implementation notes

- **Decoding a whole level** uses `jpeg-decoder` with
  `ColorTransform::None` and converts YCbCr to RGB with the fixed-point BT.601
  formula of its scalar path. Its SIMD conversion rounds the last pixels of a
  line differently, so this is the conversion a decode of part of a file can
  give the same pixels with.
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
of another threshold; in-core photographs writing nothing; the 65535 / 65536
header-only refusal in every workspace kind; the in-memory equivalent behind a
blocked mipmap directory; a cancelled build; a grey photograph; and two
threads and two processes building at once while a file is held open.
[region/tests.rs](../../../crates/sfmtool-core/src/camera/mipmap/region/tests.rs)
has the region test's cases,
[layout.rs](../../../crates/sfmtool-core/src/camera/mipmap/layout.rs) the
levels by size, and
[mipmap_options.rs](../../../crates/sfmtool-core/src/workspace/mipmap_options.rs)
each invalid `mipmap_options`.
