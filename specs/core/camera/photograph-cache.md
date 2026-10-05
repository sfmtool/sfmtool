# Photograph Cache

Many of the viewer's operations read the reconstruction's photographs as
pixels: the Image Detail panel draws one, Track View cuts patch tiles out of
several, and the bench's searches and fits measure how well a patch matches in
each image. Decoding a photograph from JPEG and building the image pyramid the
patch samplers read (the photograph plus five successively halved copies) is
the expensive part of all of them. The photograph cache holds those pyramids
once decoded, so that whichever part of the program asks next, on whichever
thread, gets the same pixels without reading the file again, and so that
memory stays under a set budget however many photographs the reconstruction
has.

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
with its five lower levels, so all 368 would be 12 GB. That is why the cache
has a byte budget, and why the callers that ask for many photographs first
leave out the ones they will not read.

## Rust API

The cache lives in
[photograph_cache.rs](../../../crates/sfmtool-core/src/camera/photograph_cache.rs),
beside `ImageU8Pyramid` in [image.rs](../../../crates/sfmtool-core/src/camera/image.rs),
exported as `sfmtool_core::camera::PhotographCache` together with
`PeekedPhotograph`, `GetManyTally` and `PhotographCacheStats`.
`default_budget_bytes` and `BUDGET_ENV_VAR` are reached through
`sfmtool_core::camera::photograph_cache`. It is in `sfmtool-core` rather than
the viewer so that `render_display_patch_bitmaps`, which `sfm web-export` also
calls, can take one.

```rust
/// Decoded photographs as pyramids, keyed by path, shared across threads.
pub struct PhotographCache { /* … */ }

impl PhotographCache {
    /// A cache holding up to `budget_bytes` of pyramids of `levels` levels.
    /// A budget of 0 keeps nothing: every call decodes and hands back a
    /// pyramid no one else will see. Panics when `levels` is 0.
    pub fn new(budget_bytes: u64, levels: usize) -> Self;
    pub fn levels(&self) -> usize;
    pub fn budget_bytes(&self) -> u64;

    /// The pyramid for `path`, decoding on a miss. Blocks while another thread
    /// is decoding the same path, and then returns its result rather than
    /// decoding a second time. `None` when the file cannot be read.
    pub fn get(&self, path: &Path) -> Option<Arc<ImageU8Pyramid>>;

    /// The pyramid for `path` if it is decoded, without reading or stat'ing
    /// anything and without waiting on a decode in flight.
    pub fn peek(&self, path: &Path) -> Option<Arc<ImageU8Pyramid>>;

    /// `peek`, telling a remembered failure apart from no answer yet.
    pub fn peek_state(&self, path: &Path) -> PeekedPhotograph;

    /// `get` for every path, the misses decoded in parallel on the rayon
    /// pool. One entry per path, in order, and a tally. Counts `images`
    /// through `progress` and polls it for cancellation before each decode.
    pub fn get_many(
        &self,
        paths: &[&Path],
        progress: &Progress<'_>,
    ) -> Result<(Vec<Option<Arc<ImageU8Pyramid>>>, GetManyTally), Cancelled>;

    /// Put pixels that did not come from this cache's decode in as `path`'s.
    pub fn insert(&self, path: &Path, pyramid: Arc<ImageU8Pyramid>);

    /// Drop one path's entry, or every entry.
    pub fn forget(&self, path: &Path);
    pub fn clear(&self);

    /// Entries, bytes held, the budget, and the hit and miss counters.
    pub fn stats(&self) -> PhotographCacheStats;
}

pub enum PeekedPhotograph { Decoded(Arc<ImageU8Pyramid>), Unreadable, NotDecoded }

pub struct GetManyTally { pub read: usize, pub reused: usize, pub unreadable: usize }

pub struct PhotographCacheStats {
    pub entries: usize, pub bytes: u64, pub budget_bytes: u64, pub hits: u64, pub misses: u64,
}

/// A quarter of physical memory, clamped to 1-16 GiB; 4 GiB when it cannot
/// be read. `BUDGET_ENV_VAR` overrides it.
pub fn default_budget_bytes() -> u64;

/// "SFM_EXPLORER_PHOTOGRAPH_CACHE_MB": a whole number of megabytes (10^6 bytes).
pub const BUDGET_ENV_VAR: &str;
```

Why it is shaped this way:

- **Keyed by path, not by image index.** An image index is a position in one
  version's image table, and an edit that deletes a camera image renumbers
  the table although no file changed. A path does not change when the table is
  renumbered, and two reconstructions over the same workspace share their
  decodes. The key is the path as joined from the workspace directory and the
  image name. It is not canonicalized, since two spellings of one file cost a
  second decode but are never wrong.
- **Taking `&self` and held in an `Arc`.** A caller holds
  `Arc<PhotographCache>` and a background job's closure captures a clone, so a
  worker puts what it decodes where the next worker and the panels find it.
- **`get` blocks, `peek` does not.** A worker wants the pixels and is prepared
  to wait for them. The GUI thread must not wait on a decode, so it asks
  whether they are there. `peek_state` adds the one thing a caller that starts
  a decode on a miss needs: whether a decode was already tried and failed, so
  it does not start another for a file that cannot be read.
- **`get_many` returns one entry per path, in order, and opens no phase.** The
  patch kernels index their view slice by image index, and the caller builds
  that slice from the entries; a `None` entry is an unreadable file. The
  caller names the phase (`decode images` in the viewer's bench steps, `decode
  photographs` in the bitmap render) and writes its note from the tally, since
  what the note should say depends on the caller.
- **A fixed pyramid depth per cache.** Every consumer that shares pixels wants
  `DISPLAY_PYRAMID_LEVELS` (6). A depth in the key would let two depths of one
  photograph both occupy the budget.
- **`insert`** exists for pixels that are not a decode of the file on disk,
  such as a test fixture standing in for a photograph. The entry records the
  file's stamp as it is now, so `get` answers with those pixels until the file
  changes.

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
    let (pyramids, tally) = cache.get_many(&refs, &phase)?;
    // … build ProjectedImage views from `pyramids` and run the kernel …
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

The cache counts the bytes of every decoded pyramid it holds
(`ImageU8Pyramid::byte_len`, the sum over its levels). When a decode takes the
total over the budget, it drops the least recently used entries until the total
is back under, never the entry just decoded. It drops only the cache's own
`Arc`, so a job holding a pyramid keeps it alive until the job ends. Eviction
is therefore safe during any read. The cost is that memory can briefly exceed
the budget by what running jobs hold, which they would have held anyway.

A `get_many` over more photographs than the budget holds evicts its own
earlier entries as it goes, and the next call finds only the tail. After the
cull, a geometry search on the test reconstruction reads 143 to 335
photographs, 5 to 11 GB, so the budget has to hold a few hundred pyramids for a
repeated search to reuse them. The first run's cost is dominated by how fast
the misses decode, which is why `get_many` decodes in parallel.

The default budget is a quarter of physical memory, clamped to between 1 GiB
and 16 GiB, or 4 GiB where physical memory cannot be read (it is read with
`GlobalMemoryStatusEx` on Windows and `sysconf` on Linux and macOS). The
environment variable `SFM_EXPLORER_PHOTOGRAPH_CACHE_MB` overrides it with a
whole number of megabytes of 10^6 bytes; `0` gives a cache that keeps nothing,
and a value that does not parse is ignored. On the 64 GB test machine the
default is 16 GiB, about 490 pyramids at 3840×2160.

## A changed file and remembered failures

An entry records the file's length and modification time when it is made.
`get` and `get_many` stat the file and decode again when either differs. A file
that cannot be read is remembered as a failure, which costs nothing against the
budget, so a missing file is not reopened on every frame; because the failure
also carries the stamp (or the absence of one), `get` retries it once the file
appears or changes. `peek` and `peek_state` do not stat, since they run every
frame, so they report a remembered failure until the next `get` notices the
file has changed. `forget` and `clear` are the other ways to retry.

## In the viewer

SfM Explorer holds one cache, `AppState::photographs` in
[state.rs](../../../crates/sfm-explorer/src/state.rs), made with
`default_budget_bytes()` and `DISPLAY_PYRAMID_LEVELS`. Every reader of the
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
- **The panels** (Image Detail, and Track View's patch tiles) never decode on
  the GUI thread. They call `display_photograph` in
  [photograph_requests.rs](../../../crates/sfm-explorer/src/state/photograph_requests.rs),
  which calls `peek_state`; on a miss it starts one `get` of that path on the
  rayon pool, records the path in `PhotographRequests` so later frames do not
  start a second, and asks egui for a repaint when the decode ends. Until then
  Image Detail says "Loading image…", and "Failed to load image" once the
  decode has failed. Under a budget of 0 nothing would be kept for the next
  frame to find, so there the photograph is decoded where it is asked for
  ([track-view.md](../../gui/track-view.md),
  [multi-panel-image-browser.md](../../gui/multi-panel-image-browser.md)).

Because the key is a path, the cache is not touched when an edit renumbers a
node's images, and it is not cleared when a node or the whole scene is closed:
reopening the same file reuses the pixels, and the budget bounds the memory
([scene-graph.md](../../gui/scene-graph.md) § "Caches",
[document-model.md](../../gui/document-model.md)).

## Implementation notes

- **Concurrency.** The map is a `Mutex<HashMap<PathBuf, Arc<Slot>>>`, where a
  slot holds a `OnceLock` for the result, the file stamp and a last-used tick.
  `get` holds the lock only to find or insert the slot, stats the file with the
  lock released, and calls `get_or_init` on the slot with the lock released.
  Two threads asking for one path decode it once, and the second waits for the
  first. Different paths decode at the same time. The byte count and the
  eviction happen under the map lock after a slot is filled, not inside the
  decode. A decode whose slot left the map meanwhile (forgotten, cleared,
  evicted or replaced by a newer file) still returns its pyramid to the caller,
  but the cache neither keeps nor counts it.
- **The decode must stay serial.** It runs inside a slot's `get_or_init`. If it
  used rayon, a rayon worker waiting in a nested join could steal another
  `get_many` item for the same path, re-enter that `OnceLock` on the same
  thread and deadlock. So the decode is `ImageU8::read_rgb` followed by
  `ImageU8Pyramid::from_image(image, levels)`, both serial, and the
  parallelism is across paths in `get_many`. For the same reason a caller
  already inside a rayon parallel loop calls `get` rather than `get_many`.
- **Counters.** A `get` that decoded nothing is a hit, whether the entry was
  there or another thread's decode was waited on; each path of a `get_many`
  counts as one `get`. Every decode is a miss, whether it succeeded or not.
  `peek` counts as a use for the eviction order but not as a hit.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| budget | `default_budget_bytes()`: physical memory / 4, clamped to 1-16 GiB; 4 GiB when unknown | bytes of pyramids held before the least recently used are dropped; [photograph_cache.rs](../../../crates/sfmtool-core/src/camera/photograph_cache.rs) |
| `SFM_EXPLORER_PHOTOGRAPH_CACHE_MB` | unset | overrides the budget, in megabytes of 10^6 bytes (`BUDGET_ENV_VAR`) |
| levels | `DISPLAY_PYRAMID_LEVELS` = 6 | pyramid depth of every entry; [display_bitmaps.rs](../../../crates/sfmtool-core/src/patch/display_bitmaps.rs), used by the viewer as `state::PYRAMID_LEVELS` |
| footprint samples | 9 × 9 | grid `view_could_see_patch` projects over the patch (`FOOTPRINT_SAMPLES` in [view_selection.rs](../../../crates/sfmtool-core/src/patch/view_selection.rs)) |

## Testing

- [photograph_cache/tests.rs](../../../crates/sfmtool-core/src/camera/photograph_cache/tests.rs):
  one decode per path under contention, least-recently-used eviction within the
  budget, the entry just decoded kept over budget, a held pyramid surviving
  eviction, remembered failures, a file that appears or is rewritten being
  decoded again, `forget` and `clear` releasing bytes, `get_many` order, tally
  and cancellation, budget 0 keeping nothing, `peek_state` telling a failure
  from no answer and not waiting on a decode in flight, `insert`, and the
  default budget from physical memory and the environment variable.
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
  (a panel's miss is decoded off the calling thread, an unreadable file ends as
  unreadable, a budget of 0 decodes on the calling thread), and
  [scene_graph/tests.rs](../../../crates/sfm-explorer/src/scene_graph/tests.rs)
  (closing everything keeps the photographs).

## Non-goals

- **Cluster patches** keeps its own reads. It wants BGR channel order and full
  pyramid depth to match `sfm cluster-patches` arithmetic, and it runs once per
  index build.
- **Display thumbnails and the 3D view's background image** keep their own
  decodes. The first make small images once per open; the second is a GPU
  texture the renderer caches.
- **No cache on disk.** Pyramids are rebuilt from the JPEGs in each session.
- **The budget is set only by the environment variable**, not in the viewer's
  UI.

## Open questions

- Whether the budget should be a setting in the viewer's UI.
- Whether the MCP surface should report the cache's `stats`, for example in
  `get_scene`, beyond the phase notes.
