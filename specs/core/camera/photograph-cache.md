# Photograph Cache

Many of the viewer's operations read the reconstruction's photographs as
pixels: the Image Detail panel draws one, Track View cuts patch tiles out of
several, and the bench's searches and fits measure how well a patch matches in
each image. Decoding a photograph from JPEG and building the image pyramid the
patch samplers read (the photograph plus successively halved copies, down to
one or two pixels on the short side) is
the expensive part of all of them. The photograph cache cuts each decoded
pyramid into photograph tiles, keeps the tiles and drops the pyramid, and
answers each photograph as a tiled photograph made of its tiles
(`TiledPhotograph`,
[view-region-pyramids.md](view-region-pyramids.md)), so that whichever part
of the program asks next, on whichever thread, gets the same pixels without
reading the file again, and so that memory stays under a set budget however
many photographs the reconstruction has. A second budget, the build budget,
bounds the whole decodes in progress on one cache at once, so a caller asking for hundreds of photographs does not
hold hundreds of decodes in memory together.

## Why it exists

The case that shaped it is *Find matches by geometry* on a 368-image
reconstruction at 3840×2160 (DaeguMuseumMasks, one camera, 65,385 points), run
on a 32-thread, 64 GB machine. Without the cache, the viewer kept only what a
panel had drawn, in a map on the GUI thread; a background step decoded every
other photograph itself, one at a time, and dropped them when it finished. The
search's `decode images` phase took 15.3 s for 362 photographs, and the same
again on a second run, while its `score views` phase took under 2 ms.

With the cache, the geometry search reads only the images that could see the
patch (see "The geometric cull" below), the misses decode in parallel, and what
a run decodes stays for the next. On the same reconstruction the `decode
images` phase takes 0.5 to 1.7 s on the first run and under 2 ms on the second,
with the same images matched. Opening that file with its patch bitmaps dropped,
which reads every photograph to render them, spends 3.95 s in `decode
photographs` the first time and 10.7 ms when the same file is opened again.

A pyramid of a 3840×2160 RGB photograph is 24.9 MB at level 0 and about 33 MB
with its eleven lower levels, and its tiles, each with a one-pixel border, are
about 0.8% more, so all 368 would be 12 GB. That is why the cache has a byte
budget, and why the callers that ask for many photographs first
leave out the ones they will not read.

## Rust API

The cache lives in
[photograph_cache.rs](../../../crates/sfmtool-core/src/camera/photograph_cache.rs),
beside `ImageU8Pyramid` in [image.rs](../../../crates/sfmtool-core/src/camera/image.rs),
exported as `sfmtool_core::camera::PhotographCache` together with
`PhotographCacheOptions`, `BuildReservation`, `GetError`, `PeekedPhotograph`,
`GetManyTally` and `PhotographCacheStats`, and the layout and tile requests'
`PeekedLayout`, `PeekedTile`, `TileError` and `TryDecode`. `default_budget_bytes`,
`default_build_budget_bytes`, `BUDGET_ENV_VAR`, `BUILD_BUDGET_ENV_VAR` and
`SLOT_WAIT_POLL` are
reached through `sfmtool_core::camera::photograph_cache`. It is in `sfmtool-core` rather than
the viewer so that `render_display_patch_bitmaps`, which `sfm web-export` also
calls, can take one.

```rust
/// Decoded photographs as photograph tiles, keyed by path, shared across
/// threads, each answered as a tiled photograph made of every one of its tiles.
pub struct PhotographCache { /* … */ }

/// The settings a cache is made with. `Default` gives `default_budget_bytes()`,
/// `default_build_budget_bytes()` and `PHOTOGRAPH_TILE_SIZE`.
pub struct PhotographCacheOptions {
    pub budget_bytes: u64,       // photograph tiles kept; 0 keeps nothing
    pub build_budget_bytes: u64, // whole decodes in progress on this cache
    pub tile_size: u32,          // the side of the tiles each decode is cut into
}

impl PhotographCache {
    /// A cache keeping up to `options.budget_bytes` of photograph tiles, cut
    /// from pyramids each of `ImageU8Pyramid::full_levels` depth for its
    /// photograph's size, with the whole decodes in progress on it bounded by
    /// `options.build_budget_bytes`. A budget of 0 keeps nothing: every call
    /// decodes and hands back tiles no one else will see.
    pub fn new(options: PhotographCacheOptions) -> Self;
    pub fn budget_bytes(&self) -> u64;

    /// Reserve up to `wanted` whole decodes of `bytes_per_build` estimated
    /// bytes each against the build budget, with one compare-exchange loop
    /// on the cache's count; never waits. Grants as many as fit, and one when
    /// none fit, so a caller always makes progress.
    pub fn reserve_builds(&self, bytes_per_build: u64, wanted: usize) -> BuildReservation<'_>;

    /// One decode, if it fits under the build budget or nothing at all is
    /// reserved: `None` otherwise, so the count passes the budget only by one
    /// decode that runs alone. Never waits.
    pub fn try_reserve_build(&self, bytes: u64) -> Option<BuildReservation<'_>>;

    /// The photograph at `path`, made of every one of its tiles, decoding it
    /// when a tile of it is not kept. Waits while another thread is decoding
    /// the same path, and then answers from its decode rather than decoding
    /// a second time, or decodes in its place if that thread stops. A decode
    /// it makes is reserved with `reserve_builds`. An error
    /// when the file does not exist, cannot be read, or has a side over
    /// `MAX_JPEG_SIDE` (65535), and `Cancelled` when `progress` is cancelled
    /// while it waits or decodes.
    pub fn get(&self, path: &Path, progress: &Progress<'_>)
        -> Result<Arc<TiledPhotograph>, GetError>;

    /// The photograph at `path` if every tile of it is kept, without reading
    /// or stat'ing anything and without waiting on a decode in flight.
    pub fn peek(&self, path: &Path) -> Option<Arc<TiledPhotograph>>;

    /// `peek`, telling a remembered failure apart from no answer yet.
    pub fn peek_state(&self, path: &Path) -> PeekedPhotograph;

    /// Whether every tile of the photograph at `path` is kept, so `peek`
    /// would answer it: reads, waits on and allocates nothing, and is no use
    /// for the eviction order.
    pub fn keeps_every_tile(&self, path: &Path) -> bool;

    /// `get` for every path, the misses decoded in parallel on the rayon
    /// pool. One entry per path, in order, and a tally. Counts `images`
    /// through `progress` and polls it for cancellation before each path and
    /// as `get` does.
    pub fn get_many(
        &self,
        paths: &[&Path],
        progress: &Progress<'_>,
    ) -> Result<(Vec<Option<Arc<TiledPhotograph>>>, GetManyTally), Cancelled>;

    /// The layout and tile requests of § "Layouts and tiles". `layout` and
    /// `peek_tile` read nothing and never wait; `request_layout` reads the
    /// header; `try_decode_tiles` is a panel's decode, which never waits;
    /// `get_tile` decodes when its tile is not kept, and
    /// `view_region_pyramid` and `get_level` read the photograph `get`
    /// answers, each decoding as `get` does. The four that can decode a whole
    /// photograph take the caller's `Progress` and poll it for cancellation
    /// as `get` does.
    pub fn layout(&self, path: &Path) -> PeekedLayout;
    pub fn request_layout(&self, path: &Path);
    pub fn peek_tile(&self, path: &Path, key: TileKey) -> PeekedTile;
    pub fn try_decode_tiles(&self, path: &Path, keys: &[TileKey], progress: &Progress<'_>)
        -> Result<TryDecode, TileError>;
    pub fn get_tile(&self, path: &Path, key: TileKey, progress: &Progress<'_>)
        -> Result<Arc<PhotographTile>, TileError>;
    pub fn view_region_pyramid(&self, path: &Path, regions: &LevelRegions,
        progress: &Progress<'_>) -> Result<ImageU8Pyramid, TileError>;
    pub fn get_level(&self, path: &Path, level: usize, progress: &Progress<'_>)
        -> Result<Arc<ImageU8>, TileError>;

    /// Put pixels that did not come from this cache's decode in as `path`'s,
    /// cut into tiles as a decode cuts them and kept as unused.
    pub fn insert(&self, path: &Path, image: ImageU8);

    /// Drop one path's entry, tiles and layout, or every entry, tile and
    /// layout.
    pub fn forget(&self, path: &Path);
    pub fn clear(&self);

    /// Tiles and bytes kept, the budget, the hit, miss and decode counters,
    /// the build budget and the bytes reserved on the build count.
    pub fn stats(&self) -> PhotographCacheStats;
}

/// Decodes reserved on one cache's build count, each of the same estimated
/// bytes: `release_one` gives one back as it ends, and dropping it gives back
/// the rest, a panic included.
pub struct BuildReservation<'a> { /* … */ }
impl BuildReservation<'_> {
    pub fn granted(&self) -> usize;
    pub fn release_one(&self);
}

pub enum GetError { NotFound(PathBuf), Unreadable(String),
                    TooLarge { path: PathBuf, width: u32, height: u32 }, Cancelled }

pub enum PeekedPhotograph { Decoded(Arc<TiledPhotograph>), Unreadable, NotDecoded }

pub struct GetManyTally { pub read: usize, pub reused: usize, pub unreadable: usize }

pub enum PeekedLayout { Known(Arc<PyramidLayout>), Unreadable(String), NotKnown }
pub enum PeekedTile { Ready(Arc<PhotographTile>), Unreadable, NotReady }
pub enum TileError { Unreadable { path: PathBuf, why: String },
                     OutOfRange { path: PathBuf, why: String }, Cancelled }
pub enum TryDecode { Decoded(Vec<Arc<PhotographTile>>), Busy }

pub struct PhotographCacheStats {
    pub tiles: usize, pub bytes: u64, pub budget_bytes: u64, pub hits: u64, pub misses: u64,
    pub decodes: u64, pub build_budget_bytes: u64, pub build_reserved_bytes: u64,
}

/// A quarter of physical memory, clamped to 1-16 GiB; 4 GiB when it cannot
/// be read. `BUDGET_ENV_VAR` overrides it.
pub fn default_budget_bytes() -> u64;

/// The same rule as `default_budget_bytes`: a quarter of physical memory,
/// clamped to 1-16 GiB; 4 GiB when it cannot be read.
/// `BUILD_BUDGET_ENV_VAR` overrides it.
pub fn default_build_budget_bytes() -> u64;

/// "SFM_EXPLORER_PHOTOGRAPH_CACHE_MB": a whole number of megabytes (10^6 bytes).
pub const BUDGET_ENV_VAR: &str;

/// "SFM_EXPLORER_BUILD_BUDGET_MB": a whole number of megabytes (10^6 bytes).
pub const BUILD_BUDGET_ENV_VAR: &str;

/// 50 ms: how often a caller waiting on another caller's decode looks at its
/// own `Progress` for cancellation.
pub const SLOT_WAIT_POLL: Duration;
```

Why it is shaped this way:

- **Keyed by path, not by image index.** An image index is a position in one
  version's image set, and an edit that deletes a camera image renumbers
  the image set although no file changed. A path does not change when the image set is
  renumbered, and two reconstructions over the same workspace share their
  decodes. The key is the path as joined from the workspace directory and the
  image name. It is not canonicalized, since two spellings of one file cost a
  second decode but are never wrong.
- **Taking `&self` and held in an `Arc`.** A caller holds
  `Arc<PhotographCache>` and a background job's closure captures a clone, so a
  worker puts what it decodes where the next worker and the panels find it.
- **`get` waits, `peek` does not.** A worker wants the pixels and is prepared
  to wait for them, as long as its own task is not cancelled: `get` takes the
  caller's `Progress` so that a cancelled task stops waiting on a decode
  another caller started, rather than wait out a decode of a second or more.
  A caller with nothing to cancel passes `&Progress::none()`. `get` reports
  nothing through it. The GUI thread must not wait on a decode, so it asks
  whether they are there. `peek_state` adds the one thing a caller that starts
  a decode on a miss needs: whether a decode was already tried and failed, so
  it does not start another for a file that cannot be read.
- **Tiles, not pyramids.** A decode is cut into photograph tiles of
  `tile_size`, each with its one-pixel border, one level at a time, and only
  the tiles are kept: the tiles a request asked for (`try_decode_tiles`'
  keys, `get_tile`'s key) as used and the rest of the decode as unused, an
  unused tile becoming used when a request reads it (`peek_tile`,
  `get_tile`). `get`, `get_many`, `peek` and `peek_state` answer a
  `TiledPhotograph` made of every tile of the photograph (`from_tiles`),
  the `Arc`s the cache keeps, so two callers of one photograph share its
  tiles and a caller holds every pixel of it while it holds the answer.
  `get` and `get_many` decode the photograph again when a tile of it is not
  kept, and `peek` and `peek_state` answer it only when every tile is. A
  kernel takes the answer as its source with no wrapping of its own.
- **`get_many` returns one entry per path, in order, and opens no phase.** The
  patch kernels index their view slice by image index, and the caller builds
  that slice from the entries; a `None` entry is an unreadable file. The
  caller names the phase (`decode images` in the viewer's bench steps, `decode
  photographs` in the bitmap render, `read photographs` in the
  cluster-patches build) and writes its note from the tally, since
  what the note should say depends on the caller.
- **Each photograph's depth set by its size.** Every entry has
  `ImageU8Pyramid::full_levels(width, height)` levels, down to one or two
  pixels on the short side: 14 for 16320×12240, 12 for 3840×2160, 9 for
  270×480. That is the depth every other builder of a photograph pyramid for
  the patch kernels uses (`sfm cluster-patches` and the Python patch
  bindings), so a kernel reads the same
  levels from a photograph whichever caller handed it over, and a footprint of
  any size finds a level: a kernel clamps its level at the pyramid's top only
  where a view shrinks the photograph to about one pixel. Depth is not a
  parameter, so the key need not hold it and two depths of one photograph never
  both occupy the budget. The levels past the sixth cost under 0.1% of a
  pyramid's bytes.
- **The build count is the cache's, and nothing waits on it.** Every caller
  that shares a cache shares its count, and two caches in one process do not
  coordinate, so how many caches a program makes decides how many such bounds
  it has: SfM Explorer holds one, `sfm web-export` makes one for its run. A
  count that blocked, like a semaphore, would make a rayon worker wait for
  another caller's permit, so `reserve_builds` grants what fits and always at
  least one decode, and `try_reserve_build`, for a caller that can ask again
  later, is refused instead.
- **`insert`** exists for pixels that are not a decode of the file on disk,
  such as a test fixture standing in for a photograph. It cuts the image into
  tiles as a decode does and keeps them as unused. The entry records the
  file's stamp as it is now, so `get` answers with those pixels until the file
  changes, or until one of the tiles is evicted, when it decodes the file.

Example, a background step in the viewer:

```rust
use std::path::{Path, PathBuf};
use std::sync::Arc;

let cache = Arc::clone(&state.photographs);
let paths: Vec<PathBuf> = needed
    .iter()
    .map(|&i| recon.workspace_dir.join(&images[i].name))
    .collect();
let job = move |progress: &Progress<'_>| {
    let phase = progress.phase("decode images");
    let refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
    let (photographs, tally) = cache.get_many(&refs, &phase)?;
    // … build ProjectedImage views from `photographs` and run the kernel …
};
```

### The geometric cull

A caller that asks for many photographs to run the patch view selection first
drops the views the selection would reject without reading pixels. The
predicate is `view_could_see_patch` in
[view_selection.rs](../../../crates/sfmtool-core/src/patch/view_selection.rs):
the patch faces the camera, its centre is in front of it, and its projected
footprint overlaps the image. `select_patch_views` tests the same predicate
before it samples a candidate, so a view it rejects is never read and a
placeholder in its slot changes nothing. How the footprint test stays
conservative is in
[patch-view-selection.md](../patch/patch-view-selection.md) § "The geometric
cull".

The viewer's geometry search reads the photographs of the track's own images
and of every image that passes the predicate for the track's patch. The cull
narrows the search less than one might hope, because it has no occlusion test
and in a capture that circles an object most cameras face most points. On the
DaeguMuseumMasks reconstruction it keeps 151 of 368 images for point 30000,
143 for point 1000 and 335 for point 50000, while the searches admit 8, 21 and
38.

## Budget and eviction

The cache counts the bytes of every tile it keeps, its pixels with their
border. When a decode, or `insert`, takes the total over the budget, it drops
tiles until the total is back under: the unused before the used, among each
the least recently asked for first, and among tiles asked for together the
finer levels first. It never drops a tile the request that made the decode
asked for, which for `get`, `get_many` and `insert` is every tile, so the
photograph just decoded stays whole even over the budget, as its caller holds
it anyway. A photograph that lost a tile keeps its layout and the tiles left,
which `peek_tile` still answers, but `peek` no longer answers it and the next
`get` decodes it again. The cache drops only its own `Arc`, so a job holding a
photograph keeps its tiles alive until the job ends. Eviction is therefore
safe during any read. The cost is that memory can briefly exceed the budget by
what running jobs hold, which they would have held anyway.

A `get_many` over more photographs than the budget holds evicts its own
earlier photographs' tiles as it goes, and the next call finds only the tail
whole. After the cull, a geometry search on the test reconstruction reads 143
to 335 photographs, 5 to 11 GB, so the budget has to hold a few hundred
photographs' tiles for a repeated search to reuse them. The first run's cost
is dominated by how fast the misses decode, which is why `get_many` decodes
in parallel.

The default budget is a quarter of physical memory, clamped to between 1 GiB
and 16 GiB, or 4 GiB where physical memory cannot be read (it is read with
`GlobalMemoryStatusEx` on Windows and `sysconf` on Linux and macOS). The
environment variable `SFM_EXPLORER_PHOTOGRAPH_CACHE_MB` overrides it with a
whole number of megabytes of 10^6 bytes; `0` gives a cache that keeps nothing,
and a value that does not parse is ignored. On the 64 GB test machine the
default is 16 GiB, the tiles of about 490 photographs at 3840×2160.

## The build budget

A whole decode holds the photograph's pixels and the tiles it cuts from them
while it runs, estimated at 5 bytes per pixel of the photograph: about 41 MB
at 3840×2160 and 1 GB at 200 megapixels. It cuts level 0's tiles, drops
the image, and makes each next level from the tiles of the one before, with
the 2×2 box filter `ImageU8::downsample_2x` applies, bit for bit, so it never
holds the whole pyramid beside its tiles: the cutting peaks at the image and
its level-0 tiles, about 2.0 times the image's bytes, and once it is cut the
tiles alone are about 1.34 times. The decoder's own peak inside `read_rgb` is
higher. Measured in a release build on an 8000×6000 JPEG (144 MB of pixels),
`get` raises the process's peak private bytes by 2.13 times the image's bytes
and keeps 1.34 times, as the decode into a whole pyramid the cache kept
before did (2.13 times, keeping 1.33), the decoder's own buffers included in
both. Each decode reserves that estimate on
the cache's build count before it reads the pixels, and gives it back when it
ends, whatever the cache's budget, 0 included. The count is an atomic in the
cache, checked against `PhotographCacheOptions::build_budget_bytes`, by
default `default_build_budget_bytes()`, worked out as the cache's budget
is: a quarter of physical memory clamped to between 1 GiB and 16 GiB, 16 GiB
on the 64 GB test machine, or 4 GiB where physical memory cannot be read. The
environment variable `SFM_EXPLORER_BUILD_BUDGET_MB` overrides it with a whole
number of megabytes of 10^6 bytes, and a value that does not parse is
ignored. A build budget of 0 lets each caller decode one photograph at a
time, since `reserve_builds` grants one decode when none fits, and
`try_reserve_build` grants a decode that does not fit when nothing at all is
reserved, so panels decode one photograph at a time too. The same rule lets a
decode larger than the whole budget, a photograph over about 214 megapixels at
the 1 GiB minimum, run alone when the count is idle rather than be refused for
ever.

- **`get`** reads the photograph's header once it has marked the slot as
  being made, whether the slot was empty or a maker that stopped has just
  emptied it, and reserves its one decode with `reserve_builds`. It holds
  nothing while it waits on another caller's decode.
- **`get_many`** first answers every path whose slot is made from the file as
  it is now, and reads the header of each other path to know its misses and
  their sizes. It then decodes the misses in groups, one `par_iter` a group,
  the groups one after another. A group is as many of the next misses as
  `reserve_builds` grants, at most the rayon pool's size and at least one,
  each reserved at its group's largest size. A member that finds its slot
  being made by another caller keeps its share while it waits, since it may
  take the decode over, and gives it back when it ends.
- **A maker that stops** (cancelled, failed or unwinding) gives its share
  back before its slot changes state, so the waiter that decodes in its place
  never finds that decode still counted, and reserves for its own decode.

Since a caller that finds the budget used up is still granted one decode, the
count goes past the budget only by one decode per caller reserving at that
moment: the bound is the build budget plus one whole decode per caller. A
caller that wants several photographs asks for them with one `get_many`, never
with `get` from many workers of one parallel loop.

## Layouts and tiles

The panels draw photograph tiles (`TileKey`, `PhotographTile` in
[tiled_photograph.rs](../../../crates/sfmtool-core/src/camera/tiled_photograph.rs)),
and a kernel can read a tile, a view-region pyramid or a level whole. The
cache answers each from the tiles it keeps, decoding the photograph when a
tile asked for is not kept, in
[photograph_cache/tiles.rs](../../../crates/sfmtool-core/src/camera/photograph_cache/tiles.rs).
A tile is `tile_size` px a side (512 by default), clipped to its level, with a
one-pixel border of its neighbours' pixels, or the edge pixel repeated at the
level's edge, so tiles drawn side by side with linear filtering show no seam.

- **`layout`** answers `Known` with the photograph's `PyramidLayout` (its
  level sizes and tile grid, in core, of `full_levels` depth) once the
  photograph is held or its header has been read, `Unreadable` with the
  message once a read has failed, and `NotKnown` otherwise. It reads
  nothing, waits on nothing and does not stat the file. The layout of a
  photograph the cache has decoded is the one its slot keeps, whether or not
  every tile is still kept, so a panel that asks every frame is answered the
  same `Arc` while the photograph's layout stays the same and nothing is
  allocated; otherwise it is what `request_layout` last read.
- **`request_layout`** reads the header and keeps the layout, or why it
  could not be read. Layouts are kept whatever the budget, a budget of 0
  included, and are not counted against it, since each is a few dozen
  bytes; `forget` and `clear` drop them.
- **`peek_tile`** answers a tile the cache keeps, without reading a file
  or waiting, and counts as a use for the eviction order, marking the tile
  as used, but not as a hit. It answers `NotReady` while the photograph is
  not decoded or the tile is not kept, and `Unreadable` for a remembered
  failure or a tile the photograph does not have.
- **`try_decode_tiles`** is a panel's decode. It answers the tiles asked
  for at once, as a hit, when the cache keeps every one of them. Otherwise it
  reads the header, refuses a key the photograph does not have as
  `TileError::OutOfRange` before it reserves anything, and reserves the
  decode with `try_reserve_build`; then it claims the slot only when no other
  caller is decoding in it, giving the reservation back if one is, and
  decodes in the slot as `get` does, keeping the tiles asked for as used and
  the rest as unused. It answers `Busy`, decoding nothing,
  when the reservation is refused or another caller holds the slot, so it
  never waits on a slot or on the build count, and panel decodes take the
  count past the build budget only by one decode that runs alone. The tiles it answers are the caller's
  under a budget of 0, where the cache keeps nothing. A file whose header
  cannot be read is remembered as its failure, as `get` remembers it.
- **`get_tile`, `view_region_pyramid` and `get_level`** are a kernel's
  reads. `get_tile` answers a tile the cache keeps, marking it used, and
  otherwise decodes as `get` does, reserving with `reserve_builds` and
  waiting while another caller decodes the photograph, and keeps the tile as
  used; `view_region_pyramid` and `get_level` read the photograph `get`
  answers.
  A level, tile or rectangle the photograph does not have, or an empty
  set of regions, is `TileError::OutOfRange`; a photograph that cannot be
  read is `TileError::Unreadable`, with `GetError`'s message.
- **Cancellation.** A whole decode can take over a second, so the four
  calls that can make one, `try_decode_tiles`, `get_tile`,
  `view_region_pyramid` and `get_level`, take the caller's `&Progress`,
  which they poll for cancellation as `get` does and report nothing
  through. A cancelled call answers `TileError::Cancelled`, which is never
  remembered: the slot is left as it was, the decode's reservation is given
  back, and the next call decodes. A panel's task passes its own `Progress`,
  or `&Progress::none()`.

## A changed file and remembered failures

Before decoding, the cache reads the photograph's size from its header and
refuses one with a side of 65536 px or more as `GetError::TooLarge`, naming
the path and the size, without decoding it, in every workspace and with no
workspace. No JPEG frame could hold such a photograph's full-resolution
[mip level](mipmaps.md#which-photographs-are-out-of-core), and one rule for
every photograph keeps its readability the same whatever its workspace sets.

An entry records the file's length and modification time when it is made.
`get` and `get_many` stat the file and decode again when either differs. A file
that cannot be read or is refused is remembered as its error, which costs
nothing against the budget, so a missing file is not reopened on every frame; because the failure
also carries the stamp (or the absence of one), `get` retries it once the file
appears or changes. `peek` and `peek_state` do not stat, since they run every
frame, so they report a remembered failure until the next `get` notices the
file has changed. `forget` and `clear` are the other ways to retry.

## In the viewer

SfM Explorer holds one cache, `AppState::photographs` in
[state.rs](../../../crates/sfm-explorer/src/state.rs), made with
`PhotographCacheOptions::default()`. Every reader of the
reconstruction's photographs goes through it except the ones under
"Non-goals":

- **Background steps** (the bench's fits, stage changes, evaluations and
  searches, *Create Track Here*, *Find Nearby Tracks*, *Add image to tracks*)
  build a `ViewSources` on the GUI thread, which holds per image the path to
  read or nothing, and a clone of the cache. The worker calls `get_many` inside
  the `decode images` phase, whose note gives how many were read from disk and
  how many reused, and how full the cache is afterwards ([edits.rs](../../../crates/sfm-explorer/src/state/edits.rs);
  [bench.md](../../gui/bench.md), [background-tasks.md](../../gui/background-tasks.md)).
- **Opening a file** with patch frames and no bitmaps renders them through
  `render_display_patch_bitmaps`, which reads every photograph with `get_many`,
  so the panels and later steps find them decoded
  ([open.rs](../../../crates/sfm-explorer/src/state/open.rs)). `sfm web-export`
  calls the same function with a cache of budget 0.
- **The cluster-patches build**, the second half of *Build Index Files*,
  reads every photograph with `get_many` in its `read photographs` phase, on
  a clone of the cache its job carries, and fails naming the first
  photograph the cache could not hand back, the error that `get` then
  answers from the remembered failure
  ([cluster_patches.rs](../../../crates/sfm-explorer/src/cluster_patches.rs);
  [index-files.md](../../gui/index-files.md)).
- **Image Detail** never reads or decodes on the GUI thread, at any budget.
  It calls `display_layout` and `display_tiles` in
  [photograph_requests.rs](../../../crates/sfm-explorer/src/state/photograph_requests.rs),
  which call `layout` and `peek_tile`; while the layout is `NotKnown` the
  first starts one `request_layout` of that path on the rayon pool, and for
  the tiles the cache does not hold the second starts one `try_decode_tiles`
  of them, each recorded in `PhotographRequests` so later frames and other
  panels do not start a second while it runs, and each asking egui for a
  repaint when it ends (50 ms later when the decode answered `Busy`, so the
  panel asks again then). The tiles a decode answers are kept in
  `PhotographRequests` until the panel asks for them, which is how a cache of
  budget 0, which keeps nothing, hands them over; each ask of Image Detail's
  for a photograph drops its kept tiles the ask does not name, a
  photograph's kept tiles are at most 32, or one decode's answer when that
  is more, an answer that would take them past 32 replacing them, and only
  the four photographs answered last are kept.
  A decode's failure is kept there too, apart from the answered tiles and
  for every photograph, with the file's length and modification time, so
  under a budget of 0 the panel says "Failed to load image" rather than
  decode again, until a check of the file on the rayon pool, started by a
  frame that asks for it, finds that the file changed; only then does the
  check ask for a repaint. The failures are not bounded as the answers are,
  since a failure dropped to make room would be decoded again by the next
  frame that shows the photograph, and panels showing more such photographs
  than the bound would decode them in turn and repaint without end. A key
  the photograph does not have, which
  only a layout read before the file changed size names, makes the decode
  `forget` the path, so the next frame reads the header again. While
  animation plays the panel asks for the next image's tiles a frame ahead
  with `prefetch_tiles`, which starts the same decode and peeks at nothing
  ([multi-panel-image-browser.md](../../gui/multi-panel-image-browser.md)
  § "Image Detail").
- **The 3D view's camera-view background** never reads or decodes on the GUI
  thread either, at any budget, and asks only when the level it wants is not
  the one its texture holds. It calls `display_layout`, and then
  `display_level` for the coarsest level that covers the viewport within the
  GPU's texture limit. `display_level` starts one task on the rayon pool,
  recorded in `PhotographRequests` with the tile decodes, so one task at most
  runs for a photograph whichever panel started it. The task asks for the
  level's tiles with `try_decode_tiles`, answering `Busy` as Image Detail's
  decode does and asked again 50 ms later, and builds the level from the
  tiles `try_decode_tiles` answered, at every budget. It makes no other call
  to the cache, and never calls `get_level`, so no task of the background
  waits on a slot or on the build count. The
  level is kept in `PhotographRequests` until the background takes it, one
  level at most, and a failure is kept and a key or level the photograph
  does not have handled as for Image Detail's decodes. The background keeps
  the level sizes, so a resize chooses the level again without asking the
  cache ([camera-views.md](../../gui/camera-views.md) § "Photograph
  background").
- **Track View's pictures**, its patch tiles, crops and their hover views,
  **the Track Table's crops** and their hover views, and **Measurement
  View's endpoint crops** and their hover views, never read or decode on the
  GUI thread either, at any budget, and no panel reads a whole pyramid. Each is
  rendered through `read_peeked` in the same file, which calls
  `display_layout` and then asks for the photograph tiles the render needs
  with `display_tile_set`: every one of them handed over together, from
  the answers kept in `PhotographRequests` or `peek_tile`, or none, while
  one `try_decode_tiles` of the missing ones runs as Image Detail's does.
  It drops no kept tile it does not ask for, since several of the panel's
  pictures read one photograph. Once all are there the render runs on the
  GUI thread on `TiledPhotograph::from_tiles` of exactly those tiles, which
  holds them for the render, so a tile the cache evicts after the peek is
  still read and the render never decodes; a render that needs tiles the
  photograph does not hold is refused with
  `PhotographReadError::NeedsTiles`, which names them, so the tiles a
  picture needs are learned from a render on the coarsest level's first
  tile, which gives the photograph's channels, and asked for together
  ([view-region-pyramids.md](view-region-pyramids.md),
  [track-view.md](../../gui/track-view.md)). While the Track Table's rows
  scroll its crops read through `read_peeked_without_asking`, which runs the
  same render on what the cache and the kept answers already hold and starts
  no read ([track-table.md](../../gui/track-table.md) § "Requests wait for
  the scroll to stop"). Measurement View cuts an endpoint's crop and its
  hover view in one render, so the tiles both need are asked for together
  ([measurement-view.md](../../gui/measurement-view.md)).
- **The image menu** asks only whether each photograph is decoded, with
  `keeps_every_tile`, and reads no pixel. The scene graph lays the menu out
  for every node on every frame, so the question allocates nothing and is no
  use for the eviction order, where `peek` would build a photograph of every
  tile and refresh every photograph's place in that order each frame:
  *Add image to tracks* is offered for an image whose photograph the cache
  holds, or whose file exists
  ([image_menu.rs](../../../crates/sfm-explorer/src/image_menu.rs)).

Because the key is a path, the cache is not touched when an edit renumbers a
node's images, and it is not cleared when a node or the whole scene is closed:
reopening the same file reuses the pixels, and the budget bounds the memory
([scene-graph.md](../../gui/scene-graph.md) § "Caches",
[document-model.md](../../gui/document-model.md)).

## Implementation notes

- **Concurrency.** The map is a `Mutex<HashMap<PathBuf, Arc<Slot>>>`, where a
  slot holds behind a mutex of its own whether one caller is making it, the
  photograph's layout once made or the remembered failure, and the tiles it
  keeps, each with whether it is used and the tick it was last asked for at;
  beside the mutex are a condition variable and the file stamp. Locks are
  taken map first: no caller that holds a slot's lock takes the map's. `get`
  holds the map lock only to find or insert the slot, and stats the file with
  it released. A caller that does not find every tile it asks for kept marks
  the slot as being made and decodes with no lock held, the tiles kept before
  staying readable meanwhile; a caller that finds it being made waits on the
  condition variable
  with `Condvar::wait_timeout` of `SLOT_WAIT_POLL`, looking at its own
  `Progress` after each wake, and stops with `GetError::Cancelled` when it is
  cancelled, leaving the decode to its maker. So two threads asking for one
  path decode it once, and the second waits for the first. Different paths
  decode at the same time. The tiles are kept, and the byte count and the
  eviction happen, under the map lock after a decode, not inside it. A tile
  the slot already keeps stays, and the decode's copy of it is dropped, so a
  caller already holding it and the cache share one. A decode whose slot left
  the map meanwhile (forgotten, cleared or replaced by a newer file) still
  returns its tiles to the caller, but the cache neither keeps nor counts
  them, and a slot that leaves the map drops its tiles.
- **Cancellation, and a maker that stops.** The maker looks at its
  `Progress` once more after the decode, before it cuts the tiles; when it
  is cancelled it cuts nothing, remembers nothing, leaves the slot as it
  found it and wakes the waiters, and the first waiter to take the slot's lock
  marks it as being made and decodes the photograph itself, while the others
  go on waiting. A slot is never left marked as being made by a caller that
  has stopped: the maker's hold on the slot is a guard that clears the mark
  and wakes the waiters when it is dropped unfinished, a panic included. The slot
  is not a `OnceLock` because a `OnceLock` cannot be abandoned once a thread
  has started to fill it, so its waiters could not stop, and a cancelled
  maker would have to finish.
- **The decode must not use rayon.** It runs while its slot is marked as
  being made. If it used rayon, a rayon worker waiting in a nested join could
  steal another `get_many` item for the same path, find that slot being made
  by its own thread, and wait on itself for ever. So the decode is
  `ImageU8::read_rgb` followed by cutting the tiles level by level
  (`cut_image` in
  [tiled_photograph.rs](../../../crates/sfmtool-core/src/camera/tiled_photograph.rs),
  which `TiledPhotograph::from_image` uses too), neither of which uses
  rayon's global pool, and the rayon parallelism is across paths in
  `get_many`.
  `read_rgb` is not always single-threaded, but nothing it starts takes work
  from the global pool, so nothing can re-enter a slot. For a JPEG wider than
  128 px, `jpeg-decoder` (built without its `rayon` feature) starts one plain
  OS thread per colour component and waits for their results before it
  returns. For a compressed OpenEXR file, the `exr` crate (whose `rayon`
  feature the `image` crate's default features turn on) builds a thread pool
  of its own for the decompression and waits on it through a channel, not a
  rayon join. For the same reason a caller
  already inside a rayon parallel loop calls `get` rather than `get_many`.
- **Counters.** A `get` that decoded nothing is a hit, whether every tile it
  asked for was kept, the failure was remembered, or another thread's decode
  was waited on; each path of a `get_many`
  counts as one `get`. Every decode started is a miss, whether it succeeded,
  failed or was cancelled, and a `get` that stopped waiting because it was
  cancelled is neither. `peek` counts as a use for the eviction order but not
  as a hit. A decode is a miss that read the photograph's pixels, so a file
  whose header cannot be read, or that is refused as too large, is a miss
  but not a decode; a maker cancelled after reading the pixels counts one.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| budget | `default_budget_bytes()`: physical memory / 4, clamped to 1-16 GiB; 4 GiB when unknown | bytes of photograph tiles kept before tiles are dropped, the unused first, then the least recently used; [photograph_cache.rs](../../../crates/sfmtool-core/src/camera/photograph_cache.rs) |
| `SFM_EXPLORER_PHOTOGRAPH_CACHE_MB` | unset | overrides the budget, in megabytes of 10^6 bytes (`BUDGET_ENV_VAR`) |
| build budget | `default_build_budget_bytes()`: physical memory / 4, clamped to 1-16 GiB; 4 GiB when unknown | estimated bytes of the whole decodes in progress on one cache at once (`PhotographCacheOptions::build_budget_bytes`) |
| `SFM_EXPLORER_BUILD_BUDGET_MB` | unset | overrides the build budget, in megabytes of 10^6 bytes (`BUILD_BUDGET_ENV_VAR`) |
| bytes per decode | 5 per pixel of the photograph | what a whole decode reserves on the build count (`WHOLE_DECODE_BYTES_PER_PIXEL`) |
| levels | `ImageU8Pyramid::full_levels(width, height)` = `floor(log2(min(width, height))) + 1`, at least 2 | depth of each photograph's tiles, from its size; [image.rs](../../../crates/sfmtool-core/src/camera/image.rs) |
| footprint samples | 9 × 9 | grid `view_could_see_patch` projects over the patch (`FOOTPRINT_SAMPLES` in [view_selection.rs](../../../crates/sfmtool-core/src/patch/view_selection.rs)) |

## Testing

- [photograph_cache/tests.rs](../../../crates/sfmtool-core/src/camera/photograph_cache/tests.rs):
  each entry at the `full_levels` depth of its own size, a view shrunk 64 times
  read at level 6 by `BilinearMip` and `Anisotropic`, one decode per path
  under contention, every caller sharing its tiles, least-recently-used
  eviction within the budget of only the tiles the budget needs, finest
  level first, and a photograph that lost one decoded again by `get`, the
  photograph just decoded kept whole over budget, a held photograph
  surviving eviction, remembered failures, a file that appears or is
  rewritten being decoded again, `forget` and `clear` releasing bytes,
  `get_many` order, tally and cancellation, budget 0 keeping nothing,
  `peek_state` telling a failure from no answer and not waiting on a decode
  in flight, `keeps_every_tile` answering before and after an eviction takes
  a tile and leaving the eviction order as it was, where `peek` changes it,
  a decode into a slot that left the map answered and not charged, `insert`
  cut as a decode is, the rule both default budgets share, from physical
  memory and each one's environment variable. The slot's tests hold a decode in progress through a hook the test build
  calls as a decode starts and as a caller starts to wait, and wait for
  every thread they start with a time limit, so a caller that never returns
  fails the test rather than hang it: a maker cancelled during its decode
  answers `Cancelled` and a caller waiting on it decodes in its place; a
  `get` and a `get_many` waiting on a held decode stop when they are
  cancelled, and the decode goes on for its maker; a maker that panics leaves
  the slot empty for the next caller, with its decode's share given back; and
  a cancelled decode is not remembered, with a budget or without one. The
  build budget's tests read the count through the same hook: `reserve_builds`
  grants what fits and one when none does, a decode larger than the whole
  budget and absurd sizes included; `try_reserve_build` never takes the count
  past the budget, from one thread or eight; a `get_many` over 20
  photographs decodes them in groups that never pass a build budget of three
  decodes, every decode reserved as it starts, with a budget and without one;
  four `get_many` callers started together on the same 20 photographs never
  reserve more than the build budget plus one decode each, an invariant that
  holds whatever order their threads run in; two caches keep separate counts,
  and a decode adds only its own reservation; and a maker cancelled
  mid-decode gives its share back, and the `get` or `get_many` that decodes in
  its place reserves, the `get` holding nothing while it waits and the
  `get_many` member keeping its share.
- [photograph_cache/tiles/tests.rs](../../../crates/sfmtool-core/src/camera/photograph_cache/tiles/tests.rs):
  a layout read from the header with nothing decoded, and every tile of
  every level, a region and every level equal to the pixels of the pyramid
  the cache decoded the photograph into before it kept tiles, each tile with
  its border as Image Detail cut it, through a panel's decode, a peek and a
  kernel's read, the photograph `peek` and `get` answer being made of the
  tiles the cache keeps, with the cache's layout and tile size; a request
  for one tile decoding the photograph once and keeping its other tiles as
  unused, which a kernel's read of another tile is answered from and which
  go first when the budget needs room, before tiles asked for earlier; a
  tile evicted and asked for again decoding its photograph again, counted
  as a miss and a decode, with the same pixels; under a budget of 0 and one
  smaller than the photograph's level 0, a panel's request answered its
  tiles, kept over the budget and answered again with no decode under the
  smaller budget; a key, level or region the photograph
  does not have refused with `OutOfRange`, a panel's decode refusing it
  before it reserves or decodes; an unreadable photograph answered and
  remembered, and `forget` and `clear` dropping a layout; the layout of a
  held photograph following its decode, and a cache of budget 0
  keeping layouts; a panel's decode refused by `try_reserve_build` decoding
  and reserving nothing and decoding once the other reservation is given
  back, with a budget and without one, and under a build budget of 0 a panel
  decode refused while another decode is reserved and granted when the
  count is idle; a panel's decode answering `Busy` at once, with no
  wait and no reservation, while another caller's decode of the same path is
  held in progress (the test bounds that thread with a time limit); one that
  loses the slot after it reserved giving the reservation back; and panel
  decodes from eight threads over 20 photographs never taking the count past
  a build budget of three decodes, with a budget and without one; and each
  of the four calls that can decode a whole photograph, cancelled during its
  decode, answering `Cancelled` with nothing reserved or remembered, the next
  call decoding, with a budget and without one; and a held photograph's
  layout answered as the same `Arc` while its size stays the same and as a new
  one once a photograph of another size is held.
- [view_selection/tests.rs](../../../crates/sfmtool-core/src/patch/view_selection/tests.rs):
  the cull on a camera behind the patch, one the patch faces away from, a
  footprint off frame or partly in frame, a point at infinity, a footprint that
  does not fully project, and a selection that is identical when the culled
  views are placeholders.
- In the viewer: [bench/tests.rs](../../../crates/sfm-explorer/src/bench/tests.rs)
  (the geometry search reads only the images that could see the patch),
  [state/edits/tests.rs](../../../crates/sfm-explorer/src/state/edits/tests.rs)
  (a second step reads nothing from disk, an unreadable photograph refuses the
  step, deleting an image keeps the decoded photographs),
  [state/open/tests.rs](../../../crates/sfm-explorer/src/state/open/tests.rs)
  (a second open reads the photographs from the cache),
  [state/photograph_requests/tests.rs](../../../crates/sfm-explorer/src/state/photograph_requests/tests.rs)
  (Image Detail's layout and tiles each read once, off the calling thread,
  with a budget and with a budget of 0, their tasks run from a queue the
  test steps on a thread of its own bounded by a time limit; a decode
  refused by the build budget ending at once while the other reservation is
  held, asked again on the next look and decoding once it is given back; a
  failed decode answered as unreadable and not decoded again until its file
  changes, five such photographs looked at in turn under a budget of 0
  neither decoded again nor repainted for; tiles answered for a view that
  moved on dropped; a key the photograph does not have reading the
  layout again; tiles asked for ahead handed over with no second decode; the
  3D background's level read once, off the calling thread, with a budget and
  with a budget of 0, built from the tiles with one call to the cache and
  equal to the level of the pyramid the cache decodes; a level read
  answering `Busy` asked again on a later frame and read once the
  reservation is given back; a level read waiting behind a
  tile decode of the same photograph, and a level not taken dropped when
  another is asked for; a level read against a layout the file no longer
  has reading the layout again; a failed level read not decoded again; a
  render from peeked tiles run once they are all there, nothing decoded on
  the calling thread, with a budget and with a budget of 0; and a render
  that asks for nothing starting no read and no check of a failed file),
  [image_detail/photograph_tiles/tests.rs](../../../crates/sfm-explorer/src/image_detail/photograph_tiles/tests.rs)
  (two panels showing one photograph ask for its layout and its tiles once,
  neither decoding as it draws, with a budget and with a budget of 0; the tiles
  asked for ahead of an image switch are the ones the next image draws; a
  switch keeps drawing the previous image until the new one's overview has
  landed, and the first image says it is loading until then),
  [scene_renderer/upload/tests.rs](../../../crates/sfm-explorer/src/scene_renderer/upload/tests.rs)
  (the camera-view background's photograph read off the calling thread and
  drawn once it lands, with a budget and with a budget of 0, and an
  unreadable one's header read once),
  [track_table_panel/tests/crop.rs](../../../crates/sfm-explorer/src/track_table_panel/tests/crop.rs)
  and [measurement_view/tests/crops.rs](../../../crates/sfm-explorer/src/measurement_view/tests/crops.rs)
  (a crop whose tiles are not there drawn as loading, its photograph asked
  for once and nothing decoded as it draws, with a budget and with a budget
  of 0 for Measurement View, and drawn once they land), and
  [scene_graph/tests.rs](../../../crates/sfm-explorer/src/scene_graph/tests.rs)
  (closing everything keeps the photographs).

## Non-goals

- **Display thumbnails** keep their own decodes. They make small images once
  per open.
- **No cache on disk.** Tiles are cut from the JPEGs in each session, and
  every answer of `get` and `get_many` holds every tile of its photograph. A
  large photograph's reduced levels can be stored as
  [mip level files](mipmaps.md), but the cache does not read them. Answers
  that read their tiles through the cache rather than hold them, and reading
  the stored levels of large ones, are proposed in
  [out-of-core-photograph-pyramid.md](../../drafts/out-of-core-photograph-pyramid.md),
  and a pyramid holding only the region of each level one render reads in
  [view-region-pyramid.md](../../drafts/view-region-pyramid.md).
- **The budget and the build budget are set only by their environment
  variables** (or a caller's `PhotographCacheOptions`), not in the viewer's
  UI.

## Open questions

- Whether the budget should be a setting in the viewer's UI.
- Whether the MCP surface should report the cache's `stats`, for example in
  `get_scene`, beyond the phase notes.
