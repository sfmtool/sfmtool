# Index Files

A reconstruction's **index files** are the two files beside its `.sfmr` that a
bench search reads: its **SIFT index**, `<stem>-sift-index.kdf`, which holds
every SIFT descriptor of the capture arranged for nearest-neighbour queries, and
its **cluster patches**, `<stem>-cluster-patches.matches`, which holds those
descriptors clustered into candidate tracks and each cluster refined into a
patch. The second is made from the first, so the viewer builds both with one
operation, *Build Index Files*, keeps each one's none / current / stale state
beside the node, and shows both in one group of the Scene tree. The index has
a spec of its own, [sift-index.md](sift-index.md); this one covers the pair,
the operation that builds them, and the cluster-patches file.

## Where the files live

The index files of `<dir>/<stem>.sfmr` are `<dir>/<stem>-sift-index.kdf` and
`<dir>/<stem>-cluster-patches.matches`: siblings of the reconstruction file,
named after its stem, so two reconstructions saved in one directory have two
pairs. Both paths are spelled in the platform's own separator, for the reason
the index's is ([sift-index.md](sift-index.md) § "Building one"). A node with no
path on disk has nowhere to put them, and every entry that would build them is
greyed with *"Save ‹label› first: the index files are written beside the .sfmr
file."*

The cluster-patches file is a `.matches` file with a clusters section and a
cluster-patches section
([`../formats/matches-file-format.md`](../formats/matches-file-format.md)). It
holds exactly what the two command-line steps write when they are run over the
node's index, which is how `scripts/track_at_pixel/dataset.py` makes one: `sfm
match --cluster`'s background-floor clustering run over the `.kdf`, then `sfm
cluster-patches`' refinement, both with their default options. Its consumers
match its images to a reconstruction's by name
(`sfmtool_core::bench::MatchesClusters`).

## What reads them

Three steps read the files, and each reads a file **only while it is
`current`**; a file in either other state is treated as absent.

- **The descriptor search** (`search_bench_track_descriptors`, Track View's
  row menu) queries the SIFT index, and is refused naming the index's own state
  when it is not current ([sift-index.md](sift-index.md)).
- **Create Track Here** (Image Detail's menu entry and its Control+Shift click,
  and `create_track_at_pixel` on the wire) reads both: the cluster patches for
  the track-at-pixel cascade's clusters member, and the SIFT index, with every
  image's `.sift` keypoints, for its constellation member
  ([bench.md](bench.md) § "Create Track Here"). It is not refused when they are
  missing or stale, because its transfer and sweep members read neither: the
  member that would have read a file refuses and names what it lacked, and when
  every member refuses, the refusal says which files were not current and names
  *Build Index Files*. The contents are read on the worker, the clusters out of
  the `.matches` and the keypoints out of the `.sift` files; what the node holds
  open is the forest handle and the file's facts, as below.
- **Resect Image** (the image menu's entry, and `resect_camera_image` on the
  wire) reads the cluster patches, whose clusters it uses beside the tracks as
  correspondences, each cluster as a track of its own
  ([edits/resect-image.md](edits/resect-image.md)). It **needs** the file: when
  the file is `none` or `stale`, the entry is greyed and the step refused with
  one sentence that names the state (and a stale file's reason) and ends with
  *Build Index Files* / *Rebuild Index Files*, or with *save first* on a node
  with no path. It reads the whole `.matches` on every run, on the GUI thread,
  as the step itself runs there. The SIFT index is not read.

## none, current, stale

Each file is in one of three states, with the words the index uses:

| State | Meaning |
|-------|---------|
| `none` | No file at the node's path for it, and none opened by hand. |
| `current` | The file answers for this node as it stands. |
| `stale` | Anything else, with a sentence naming the first discrepancy. |

What makes the index current is [sift-index.md](sift-index.md) § "none,
current, stale": its image table is the node's, row for row, and every image's
`.sift` content hash is the one on disk now.

### When the cluster patches are stale

The cluster patches are current when all of these hold, and stale naming the
first that does not:

1. The file has both the clusters and the cluster-patches sections. A file with
   clusters and no refinement holds clusters that were never refined.
2. Its image table has as many rows as the node has images.
3. Row `i` of its image table names the node's image `i`. A cluster member
   names its image by row, so the rows have to be the node's, in its order, as
   the index's do.
4. It records the content hash of the index it was made from, under
   `matching_options.index_content_xxh128`.
5. An index is open beside the node.
6. That index's content hash is the one the file records.
7. That index is current.

The last four are the rule that ties the pair together: **a cluster-patches
file is only as current as the index it was made from.** The file records which
index that was by the `.kdf`'s whole-content hash
(`KdfFile::content_xxh128`), which covers every section of the index, so a
recorded hash equal to the open index's means the clusters were made from the
descriptors that index holds. Whether those descriptors are still the ones in
the `.sift` files is the index's own third test, which the cluster patches
inherit through test 7 rather than read the `.sift` files a second time. So
features extracted again make both files stale; a second index opened in place
of the first makes the cluster patches stale and leaves the index current; and
an index that is closed or missing leaves nothing to say what the clusters were
made from.

A cluster-patches file written by the command-line steps records the index's
file name under `index` and no content hash, so it reads stale by test 4 when
opened here; a build replaces it with one that records both.

### When the question is asked

The cluster patches' state is derived when the file is opened or built, and
again when either of two things has moved since: the node's image table (a
version that adds, removes, renames or reorders an image), or the index open
beside the node (a different path, a different content hash, or a change
between current and stale). The index's open, build, close and re-derivation
each ask the cluster patches again. The facts the verdict needs, the file's
image names and the hash it records, are read when the file opens, so asking
again reads no file. It is never asked per frame, and a file changed on disk
under a running viewer is found by opening it again or rebuilding.

## Opening on sight

`AppState::refresh_index_files` looks for the index and then the cluster
patches at the node's own paths, when the Scene tree draws, when Track View
draws, and when the first item goes onto the node's bench. The index opens
without decoding a tree or a descriptor block; the cluster patches are read for
their metadata and image names only, two JSON entries of the archive
(`sfmtool_matches_format::read_matches_image_names`). Each look is remembered
per node, the miss included, so a node with neither file is not stat-ed every
frame. Nothing is built behind the person's back, and the rows the lazy open
writes are attributed to the viewer, for the reason the index's are.

A file that does not open at all is not a fourth state: the node reads `none`
for it, and the Action Log carries one row naming the file and what the reader
said about it.

## Building them

*Build Index Files* is one background task, `Build index files`
([background-tasks.md](background-tasks.md)), that writes the index and then
the cluster patches from it, each at the node's own path, replacing what is
there, and opens both. It is live on a node that has been saved, whose images
have at least one `.sift` file, and that nothing is running on
(`AppState::build_index_files_refusal`).

**A current index is kept when the cluster patches are the file that is
missing or out of date.** When the index open beside the node is at the node's
own index path and current, and the cluster patches are not current, the build
skips the index and makes the cluster patches from the index that is open, and
its Action Log row says the index was already current. Every other build
writes both, so a node whose two files are current rebuilds both, and a node
whose index is stale gets cluster patches made from the index the same build
has just written. The entry reads *Build Index Files* when the node has
neither file open and *Rebuild Index Files* when it has either
(`AppState::index_files_build_label`).

### The phases

The index half is [sift-index.md](sift-index.md) § "Building one": `read
descriptors`, `build forest`, `write index`. The cluster-patches half is five
more:

| Phase | What it does |
|-------|--------------|
| `count features` | Reads each image's `.sift` metadata: its feature count, its image size, its two hashes, as `sfm match --cluster` records them. The counts give each image's offsets in the index, and their sum is checked against the index's length. |
| `cluster features` | The background-floor clustering of `sfm match --cluster` over the index: a self-join of every descriptor at `d + 1 = 11` neighbours and 128 leaf checks, then the clustering with `d = 10`, `alpha = 0.8`, `min_size = 2` (`sfmtool_core::features::cluster_match::background_floor_clusters_lazy`, the function the `background_floor_clusters_kdf` binding calls). |
| `read photographs` | Decodes every photograph into a full pyramid, in OpenCV's blue-green-red channel order. |
| `refine patches` | `sfm cluster-patches`' refinement with its defaults: a 12-unit patch, 25 samples a side, ZNCC 0.85, 3 px of shift, 0.35 keypoint uncertainty (`sfmtool_core::patch::cluster_refine::refine_cluster_patches`), then the warp-consistency residuals. |
| `write cluster patches` | Writes the `.matches`, through a temporary sibling renamed over the target. |

The members' detected positions and shapes are read out of the index's feature
geometry, which holds the `.sift` values bit for bit, rather than from the
`.sift` files; the refinement is handed each image's detections at their own
rows and zeros elsewhere, which is how `sfm cluster-patches` hands them. The
written geometry is the refinement's for the members it measured and the
detection for the rest, as that command writes it. The file records
`matching_options` of `mode`, `d`, `alpha`, `min_size`, `max_leaf_checks`, the
index's file name as `index`, and its content hash as `index_content_xxh128`.

**Parity with the command line.** Over the same index and the same decoded
pixels the file is the command-line steps' file array for array, down to the
bits of every float. What can differ is the decode: the viewer reads JPEG with
the `image` crate and the command line with OpenCV, and the two decoders
disagree by up to 3 levels on about 1% of the pixels of a seoul_bull
photograph. That leaves the clusters identical and moves the refinement: on
seoul_bull, 40 of 12,629 member statuses and about half the member ZNCC values
differ. With the viewer's decoded pixels handed to the command-line step
instead of OpenCV's, every array is identical.

### Progress and stopping

The two halves take an eighth and seven eighths of the bar when both are
built, and the cluster half takes the whole bar when the index is kept; inside
the cluster half the five phases take 1, 32, 4, 24 and 3 sixty-fourths. On the
17-image seoul_bull capture the build is 1.1 s, 0.09 s of it the index, 0.58 s
the self-join and clustering and 0.36 s the refinement. The self-join counts the
descriptors it has answered, the metadata read and the decode count images,
and the refinement counts clusters, a batch of 256 at a time.

**Every phase but the write stops.** The self-join reads the flag in front of
every query, the metadata read and the decode in front of every image, and the
refinement between batches; each cluster's refinement is self-contained, so
the batches change the schedule and not the answers. A build that stops or
fails in its cluster half after writing its index still hands the index back,
so the node opens the file that is now on disk; the Action Log row says the
build was cancelled after it wrote the index, or that it built the index and
could not build the cluster patches, naming why. A photograph that cannot be
decoded fails the cluster half naming its file. The cluster patches that were
there before are left as they were, because the write is the last step and
renames over the target only once the file is whole.

## The Scene tree rows

The node's last group row is **Index Files**, open by default, with the two
files under it ([scene-graph.md](scene-graph.md) § "Tree rows"):

```
▾   Index Files     2 of 2 current
      SIFT Index       37,167 descriptors
      Cluster Patches  5,117 clusters
```

The group row counts how many of the two are current, in the warning colour
when either is stale and dimmed when neither is there. Each child reads its
size when current, `stale` in the warning colour, `none` dimmed, and
`building...` while the build runs; its hover carries its file, its counts, and
the sentence naming the first discrepancy, or where a build would write it. The
group row's menu carries *Build Index Files* and *Close Index Files*; each
child's carries those two with an *Open...* for its own kind of file between
them, a `.kdf` for the index and a `.matches` for the cluster patches. A file
opened by hand is adopted whatever its state, and its row says why a stale one
will not do.

## The interface

The pair is in [index_files.rs](../../crates/sfm-explorer/src/index_files.rs),
the cluster patches in
[cluster_patches.rs](../../crates/sfm-explorer/src/cluster_patches.rs), and the
index in [sift_index.rs](../../crates/sfm-explorer/src/sift_index.rs), each file
held per node on `AppState`.

```rust
/// The state word both files use, on the rows and on the wire.
pub(crate) enum IndexFileState { None, Current, Stale }

pub(crate) const BUILD_INDEX_FILES: &str = "Build Index Files";
pub(crate) const REBUILD_INDEX_FILES: &str = "Rebuild Index Files";

/// Where `node`'s cluster patches go: `<stem>-cluster-patches.matches`.
pub(crate) fn cluster_patches_path(node: &SceneNode) -> Option<PathBuf>;

impl AppState {
    /// The look and the re-derivation for both files, in that order.
    pub(crate) fn refresh_index_files(&mut self, id: ReconId);
    pub(crate) fn build_index_files_refusal(&self, id: ReconId) -> Option<String>;
    pub(crate) fn index_files_build_label(&self, id: ReconId) -> &'static str;
    pub(crate) fn start_build_index_files(&mut self, id: ReconId) -> Result<(), String>;
    /// The build's work, owning what it reads, for the starter and the tests.
    pub(crate) fn build_index_files_job(&self, id: ReconId) -> Result<Job, String>;
    /// Opens the named files, or the node's own where none is named.
    pub(crate) fn open_index_files(&mut self, id: ReconId, sift_index: Option<PathBuf>,
                                    cluster_patches: Option<PathBuf>) -> Result<(), String>;
    pub(crate) fn close_index_files(&mut self, id: ReconId) -> Result<(), String>;

    pub(crate) fn cluster_patches(&self, id: ReconId) -> Option<&ClusterPatches>;
    pub(crate) fn cluster_patches_state(&self, id: ReconId) -> IndexFileState;
    pub(crate) fn cluster_patches_path(&self, id: ReconId) -> Option<PathBuf>;
}
```

`open_index_files` requires a named file to open and refuses the step when it
does not; a file that is not named is opened when it is at the node's own path
and reads `none` otherwise, and the step is refused when neither file opened.
Opening again is how a person asks about files changed on disk. Closing lets go
of both and remembers the misses, so the next frame does not open them again.
None of the three, and not the build, is a version.

## The wire

The operation and its files have one set of names on the wire
([mcp-server.md](mcp-server.md)):

- `build_index_files` `{ reconstruction_label }` starts the build on a worker
  and answers as `evaluate_bench_track` does.
- `open_index_files` `{ reconstruction_label, sift_index_path?,
  cluster_patches_path? }` opens the files by the rule above.
- `close_index_files` `{ reconstruction_label }` lets go of both.

`get_bench` and each node of `get_scene` carry the `index_files` object, one
entry per file:

```json
{
  "sift_index": {
    "state": "current", "path": "/runs/demo-sift-index.kdf",
    "descriptors": 37167, "images": 17, "stale_reason": null
  },
  "cluster_patches": {
    "state": "current", "path": "/runs/demo-cluster-patches.matches",
    "clusters": 5117, "members": 12629, "images": 17, "stale_reason": null
  }
}
```

`path` is the file that is open, or the node's own path for it when none is;
the counts are `null` when nothing is open.

## Testing

[cluster_patches/tests.rs](../../crates/sfm-explorer/src/cluster_patches/tests.rs)
and [index_files/tests.rs](../../crates/sfm-explorer/src/index_files/tests.rs),
headless over the index's workspace fixture, which writes a `.sift` file and a
photograph per image. Covered: the cluster-patches path being the `.sfmr`'s
stem beside it in one convention, and absent on an unsaved node; a build
writing a file with both sections, the node's images in its order and the
index's content hash, that reads current; **the file equalling, array for array
and bit for bit, the two command-line steps re-derived independently from the
same index and photographs** -- the self-join and clustering as the binding ran
them, the detections read from the `.sift` files, the refinement in one call
with the options `sfm cluster-patches` passes; each staleness cause with its
own sentence -- a stale index, another index open, no index open, a version
that removes an image, a file over other images, a clusters file with no
patches section; the look at an absent file being silent and remembered, and a
later session opening both files on sight with rows of the viewer's own. The
build: its label reading *Build* then *Rebuild*; one build making both files
current; a current index kept, byte for byte, while the missing cluster patches
are made; a stale index rebuilding both; fractions in both halves under all
eight phases, ending at the end; a cancel in the cluster half handing back the
index it wrote and writing no cluster patches; a missing photograph failing the
cluster half naming the file; and the open and the close refusing when there is
nothing to do. The rows are tested in
[scene_graph/tests.rs](../../crates/sfm-explorer/src/scene_graph/tests.rs), the
wire in [mcp/tests/render.rs](../../crates/sfm-explorer/src/mcp/tests/render.rs)
(the `index_files` object on `get_bench` and `get_scene`, the three tools, and
a build over a node whose cluster patches are missing keeping its index).

## Non-goals

The viewer does not watch the filesystem, and does not verify a file's contents
beyond what its state needs. It does not decode photographs through OpenCV, so
a JPEG capture's refinement can differ from the command line's by what the two
decoders differ by (§ "Building them"). A build writes the files at the node's
own paths only; a pair under other names is made outside the viewer and opened
with *Open...* or `open_index_files`.
