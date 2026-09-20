# The SIFT Index

A reconstruction's **SIFT index** is one file holding every SIFT descriptor its
photographs contain, arranged so that "which other photographs contain this
patch" can be answered in milliseconds rather than by comparing the patch
against every descriptor of the capture in turn. The viewer builds one on
request, keeps it open beside the reconstruction it belongs to, and queries it
when a person asks a track on the bench to find the images it is missing. It is
a `.kdf` file -- a randomized k-d tree forest
([`../core/features/randomized-kdtree-forest.md`](../core/features/randomized-kdtree-forest.md))
-- written next to the `.sfmr` it indexes.

The index is a fact about a **reconstruction**, not about a panel or a
workspace: every track on a node's bench queries the same forest, a match it
returns names an image by its position in the index's own image table, and the
candidate that match becomes names an image by its position in the
reconstruction's. Those two numbers have to be the same number, which is what
makes "is this index still an index of this reconstruction" a question the
viewer has to be able to answer at any moment, and the reason a node carries
one index rather than a workspace carrying one.

## Where the file lives

The index of `<dir>/<stem>.sfmr` is `<dir>/<stem>-sift-index.kdf`. It is the
reconstruction file's sibling and takes its name from it, so two
reconstructions saved in one directory have two indexes, and a reconstruction
whose images span several image directories has one index in one obvious place.

A node with no path on disk -- the demo scene, a value never saved -- has
nowhere to put one. Every entry that would build an index for it is greyed with
*"Save ‹label› first: the SIFT index is written beside the .sfmr file."*

A reconstruction saved under a new name leaves its index behind under the old
one. *Open...* adopts it, and it reads as current if it still is.

## none, current, stale

An open index is in one of two states, and a node with no index is in a third.

| State | Meaning | A bench search |
|-------|---------|----------------|
| `none` | No file at the node's index path, and none opened by hand. | Refused; the entry offers the build |
| `current` | The index is over exactly this node's images, in this node's order, from the `.sift` files that are on disk now. | Runs |
| `stale` | Anything else. | Refused; the entry offers the rebuild |

The index is current when all of these hold, and stale naming the first that
does not:

1. The index's image table has as many rows as the node has images.
2. Row `i` of the index names the node's image `i`.
3. For every image, the `.sift` content hash the index recorded equals the
   content hash of the `.sift` file the node resolves for that image now. An
   image with no `.sift` file matches a row that recorded none.

**Every discrepancy is stale, an index over a superset of the node's images
included.** A forest query returns its nearest descriptors across the whole
corpus. Hits from images the node no longer has would be dropped after the
fact, and they would already have taken the places of hits from images it does
have, so the answer differs from the one an index over the node's own images
gives. There is no partial state for the same reason in the other direction: an
index missing some images answers a different question than the one asked.

The third test reads the content hash from each `.sift` file's metadata and
decodes no feature. It compares against the file **on disk** and not against
hashes held in the reconstruction, for two reasons: an `embedded_patches`
reconstruction carries image hashes and no `.sift` hashes, and a search reads
its query keypoints from the `.sift` on disk, so the file on disk is the thing
the index has to agree with.

A stale index **stays open and stays named**, so a person sees what is there
and why it will not do. *Open...* on a file that is stale for this node opens it
in that state and says so in the Action Log, rather than refusing.

### When the question is asked

The state is derived when an index is opened or built, and again when a version
lands on the node whose image table differs from the one it was derived
against. It is held per node and **never computed per frame**: a version that
moves geometry, poses or the bench leaves the corpus's claim about the node
exactly as true as it was, and only one that adds, removes, renames or reorders
an image can change the answer.

It is **not** re-derived when a `.sift` file changes on disk under a running
viewer. *Rebuild* and re-opening the file are the two ways to ask again.

## The Scene tree row

Each reconstruction node carries a **SIFT Index** row among its group rows,
last, after Points and after Patches where that row is present
([`scene-graph.md`](scene-graph.md) § "Tree rows"). It is the one part of a node
that is a file beside it rather than something the node holds.

```
  SIFT Index   1.2M descriptors          current
  SIFT Index   stale                     warning colour
  SIFT Index   none                      dimmed
```

- Hover text: the index path; when one is open, the descriptor and image
  counts; when stale, the sentence naming the first discrepancy (*"‹file›
  indexes 15 images and this reconstruction has 13."*, *"The features of
  ‹image› were extracted again after this index was built."*).
- The row selects nothing and has no children, and carries no eye: nothing here
  is drawn in the viewport. It follows the tree's other conventions -- fixed
  height, an explicit id, one click target across the row.
- Its context menu carries *Build SIFT Index* (reading *Rebuild SIFT Index*
  when one is open), *Open...*, and *Close Index* (live only when one is open).
  Each is greyed with its own sentence while the node is busy.
- While the build runs the row reads `building...`; progress and cancellation
  are the background task's own
  ([`background-tasks.md`](background-tasks.md)).

The reconstruction row's context menu carries *Build SIFT Index* (*Rebuild SIFT
Index* when one is open) too, above *Convert to Embedded Patches*, under the
same gate. It is where a person looks first.

## The search entry

In the Track Edit panel a row's context menu offers *Search for matching
features* when the node's index is current ([`track-edit.md`](track-edit.md)
§ "Right-clicking a row"). When it is not, the entry itself is the remedy:

- `none`: the entry reads *Build SIFT Index to Search* and starts the build.
- `stale`: the entry reads *Rebuild SIFT Index to Search*, starts the build,
  and carries the staleness sentence as its hover text.
- When the build cannot start -- a busy node, no `.sift` files, no path on disk
  -- the entry is greyed with that refusal.

The build does **not** run the search when it finishes. A build over a large
capture takes long enough that the person has moved on, and a search that lands
candidates on a track unasked is a surprise.

## Opening on sight

The node's index path is looked at when the Scene tree draws a row, when the
Track Edit panel draws, and when the first item goes onto the node's bench.
Opening a `.kdf` decodes no tree and no descriptor block
([`../core/features/lazy-kdforest-query.md`](../core/features/lazy-kdforest-query.md)),
so looking costs a stat and a header read, and a session finds what the last one
built. The look is remembered per node, including the miss, so a reconstruction
with no index beside it is not stat-ed once a frame.

**Nothing is built behind the person's back.** An index that is not there is the
ordinary state of a reconstruction, and the row says `none` rather than refusing
anything. The row the lazy open writes is attributed to the **viewer** rather
than to whoever was acting: nobody asked for it, and it lands in the middle of
the step that set it off, so attributing it to that step's actor would make it
the last row the step wrote -- and a wire reply that reports the step's own
sentence would report this one instead
([`action-log.md`](action-log.md)).

## Building one

*Build* is a background task over every `.sift` file the node's images resolve
to, reporting the phases `read descriptors`, `build forest` and `write index`.
It writes the node's index path, replacing what is there, and opens what it
wrote.

**All three phases report, and all three stop.** The three are roughly a
quarter, a quarter and a half of the time on a 370-image capture, and each moves
the bar within its own share: the read per image, the forest per leaf placed
across its trees, and the write per batch of blocks weighted by where the
write's own time goes ([`../formats/kdf-file-format.md`](../formats/kdf-file-format.md)).
Per leaf rather than per tree, because the trees are built in parallel and four
of them are four steps that all land at the end. The same three places are where
the build reads the cancel flag, so *Cancel* stops it within a block rather than
at the end of a phase ([`background-tasks.md`](background-tasks.md)).

**A build writes beside its target and renames over it at the end.** The file it
streams into is the target's name with `.building` on it, in the target's own
directory, which makes the last step a rename rather than a copy across
filesystems; the rename replaces what is there. Until that instant the index
that is open stays exactly as it was, so a rebuild that is cancelled ten seconds
in, or that fails on its last block, leaves the working index standing rather
than replacing it with nothing. A cancelled or failed build removes what it was
writing into.

The corpus it writes carries **one image-table row per image of the node**, in
the node's own order, including images with no `.sift` file -- an image with no
features contributes no descriptor and still takes its row, which is what keeps
a corpus image index and a node image index the same number. Each row records
the image's feature tool hash and `.sift` content hash, read off the very
archive the descriptors came out of, so an index built here and untouched since
reads as current; a row for an image with no `.sift` file records zeros.

**A build may be told where to write, inside the `.sfmr`'s own directory.** The
wire's `build_sift_index` takes an optional `path`, because a second index over
the same capture under a name of its own is a reasonable thing for an agent to
ask for. A path that resolves outside that directory is refused naming it: an
index is written beside the reconstruction it indexes, and a step that took a
string and wrote wherever it pointed would be a different kind of tool. A
relative path is resolved against that directory and `.` and `..` are folded
lexically -- the file is not there yet, so there is nothing to canonicalize.

**The path is spelled in one convention.** A `.sfmr` path the session was handed
can be spelled with either separator, and a name joined onto it comes out as
`…\runs/demo-sift-index.kdf`: it opens, and it reads in the row, the reply and
the log as two conventions arguing. The index path is rebuilt from its
components where it is formed, so what the tree, the wire and the Action Log say
is the platform's own spelling throughout.

Neither building nor opening nor closing is a version. An index is a file beside
the `.sfmr` and a handle on it; the reconstruction and the bench are untouched,
so there is nothing for Undo to take back, and what each writes is one Action Log
row of kind `Bench`.

## The interface

The index lives in
[sift_index.rs](../../crates/sfm-explorer/src/sift_index.rs), one open forest
per loaded node on `AppState`. The Scene tree's row and its menu are in
[scene_graph/](../../crates/sfm-explorer/src/scene_graph/), the search entry in
[track_edit/table.rs](../../crates/sfm-explorer/src/track_edit/table.rs), and
the wire in [mcp/bench.rs](../../crates/sfm-explorer/src/mcp/bench.rs).

```rust
/// One node's open SIFT index, and what the node makes of it.
pub(crate) struct SiftIndex {
    pub(crate) path: PathBuf,
    pub(crate) forest: Arc<LazyKdForestU8>,
    pub(crate) images: usize,
    /* the staleness verdict, and what it was derived against */
}

impl SiftIndex {
    pub(crate) fn feature_count(&self) -> usize;
    pub(crate) fn stale_reason(&self) -> Option<&str>;
}

pub(crate) enum SiftIndexState { None, Current, Stale }

/// Where `node`'s index goes: `<stem>-sift-index.kdf` beside its `.sfmr`.
pub(crate) fn index_path(node: &SceneNode) -> Option<PathBuf>;

impl AppState {
    pub(crate) fn sift_index(&self, id: ReconId) -> Option<&SiftIndex>;
    pub(crate) fn sift_index_state(&self, id: ReconId) -> SiftIndexState;
    pub(crate) fn sift_index_path(&self, id: ReconId) -> Option<PathBuf>;
    pub(crate) fn building_sift_index(&self, id: ReconId) -> bool;

    /// Open the node's index if nothing has looked yet, and re-derive the
    /// state when a version has moved the image table.
    pub(crate) fn refresh_sift_index(&mut self, id: ReconId);
    pub(crate) fn open_sift_index(&mut self, id: ReconId, path: Option<PathBuf>)
        -> Result<PathBuf, String>;
    pub(crate) fn close_sift_index(&mut self, id: ReconId) -> Result<(), String>;
    pub(crate) fn start_build_sift_index(&mut self, id: ReconId, path: Option<PathBuf>)
        -> Result<(), String>;

    /// The build's work on its own, owning everything it reads, for the
    /// starter above and for a test that drives it directly.
    pub(crate) fn build_sift_index_job(&self, id: ReconId, path: Option<PathBuf>)
        -> Result<Job, String>;

    /// Why a search cannot query this node's index, or `None` when it can.
    pub(crate) fn sift_index_search_refusal(&self, id: ReconId) -> Option<String>;
    pub(crate) fn build_sift_index_refusal(&self, id: ReconId) -> Option<String>;
}
```

`refresh_sift_index` is the one entry point a panel calls: the look and the
re-derivation are the same question asked at the same moments, and a caller that
had to remember both would eventually forget one. The staleness verdict is a
`String` held on the index rather than an error returned at open, because a
stale index is a thing the viewer goes on showing.

`sift_index_search_refusal` is the single gate a search passes: the greyed
entry, the step and the wire refusal are one sentence, so a caller that asks
anyway is told what the menu already said.

## The wire

`get_bench` reports the index under `sift_index`, and each node of the
`get_scene` reply carries the same object
([`mcp-server.md`](mcp-server.md)):

```json
{
  "state": "current",
  "path": "/runs/demo-sift-index.kdf",
  "descriptors": 1204551,
  "images": 243,
  "stale_reason": null
}
```

`path` is the file that is open, or the node's own index path when none is, so
an agent can see where a build would put one; it is `null` on a node with no
path on disk. `descriptors` and `images` are `null` when nothing is open.

Three tools act on it: `open_sift_index` adopts a `.kdf`, `build_sift_index`
makes one out of the node's `.sift` files on a worker thread, and
`close_sift_index` lets go of what is open. None of them pushes a version.
`search_bench_track_descriptors` needs `state: "current"` and is refused with
the staleness sentence otherwise.

## Testing

[sift_index/tests.rs](../../crates/sfm-explorer/src/sift_index/tests.rs),
headless over a temporary directory holding a `.sfmr` path, a `.sift` file per
image and a real built `.kdf`. Covered: the index path being the `.sfmr`'s stem
beside it and spelled with one kind of separator; the refusal on a node with no
path on disk, and on one whose images have no `.sift` companion; a build writing
a caller's path inside the `.sfmr`'s directory and opening what it wrote, and
refusing one outside it or one that climbs out with `..`; a built index reading
as current over both an `embedded_patches` and a `sift_files` node; each
staleness cause producing its own sentence -- a row count, a name out of place,
a re-extracted `.sift`, one that appeared, one that vanished -- with a superset
index stale; a version that moves the image table re-deriving the state and one
that does not leaving it alone; the look at an absent file being silent and
remembered; and closing letting go of the forest, leaving the file and refusing
the search. Two more drive the build's job directly, over a `Progress` that
records what it reports: the fractions climbing past the read's share and past
the write's, so a bar that only moved through the read would fail; and a
cancelled rebuild leaving the index that is there byte for byte as it was, with
no `.building` file beside it.

The Scene tree row's three texts and its menu are in
[scene_graph/tests.rs](../../crates/sfm-explorer/src/scene_graph/tests.rs),
through whole headless frames: the row saying `none`, counting a current
index's descriptors and reading `stale`; its menu carrying the build, the open
and the close; the reconstruction row's menu carrying the build above *Convert
to Embedded Patches*; and an unsaved node's build entry greyed.

The search entry's three labels are in
[track_edit/tests.rs](../../crates/sfm-explorer/src/track_edit/tests.rs), with
the assertion that choosing a build label asks for the build and asks for no
search, and that the panel draws no index row of its own.

The wire is in [mcp/tests.rs](../../crates/sfm-explorer/src/mcp/tests.rs): the
`sift_index` object under `get_bench`, the three tools parsing and reaching the
viewer, and the search refused with the state's own sentence.

## Non-goals

The viewer does not verify that an index's descriptors are the bytes the `.sift`
files hold; it compares the recorded content hash and stops there, which is what
makes the check cheap enough to run at every open. A full verification is the
`.kdf` format's own
([`../formats/kdf-file-format.md`](../formats/kdf-file-format.md)).

The viewer does not watch the filesystem. A `.sift` file rewritten under a
running viewer leaves the index reading current until the person rebuilds it or
opens it again.
