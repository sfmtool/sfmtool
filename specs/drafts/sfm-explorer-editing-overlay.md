# Editing an in-memory reconstruction: what the overlay leaves open

**Status:** Draft

Answers Part 4 of [`sfm-explorer-editing.md`](sfm-explorer-editing.md), the
umbrella draft for editing a loaded reconstruction in place, and holds what is
left of that answer after the core landed.

The representation itself is built and standing:
[`../core/reconstruction/edited-reconstruction.md`](../core/reconstruction/edited-reconstruction.md)
describes the base plus its edits, the delete-and-re-add rule for point edits,
the stable indexes, the accessor that reads one point without materialising,
the materialisation that keeps every point in its place with its row map, and
the base and point-edit hashes. This draft keeps only what that spec does not
decide: how the GPU side consumes an overlay, what a save of an addition means
on a `sift_files` reconstruction, the point-id version graph with its minting
and resolution rules, and the open questions.

---

## The GPU side

Built and standing: [`../gui/document-model.md`](../gui/document-model.md),
"Change detection by identity". The base's buffers keep their identity, so the
upload sees no change on the base and uploads nothing for it, and the deleted
set reaches the point and patch shaders as a per-instance mask written only
where the set moved.

What that leaves open is the half an addition needs. The additions upload as a
second instance buffer drawn after the base's, with the same per-node uniforms.
The patch atlas is the one piece of GPU state the stable-index rule does not
already cover: its slot assignment is a compaction over the points that carry a
bitmap, so a point's atlas slot is not its index, and a point edit that changes
a bitmap needs the additions' patches in a second atlas, or a slot map the
base's atlas keeps across edits. The bundle already keeps a point-index-to-slot
map for the mask, which is the smaller half of that question. A materialisation
replaces both buffers with one, through the row map. No edit re-uploads a
million points.

## The file side

A save materialises, then writes as any writer does. An observation added from
a clicked pixel has a keypoint and a patch but no feature index, so the first
track edits are built on `embedded_patches` reconstructions, where that is
what an observation is. On a `sift_files` reconstruction such an edit refuses
until the format carries an observation without a feature; that format
decision is not made here.

---

## Point ids

A point id today is a coordinate in a file: `pt3d_{hash}_{index}`, the first
eight hex digits of the file's `content_xxh128` and the point's row in it
([the format spec's Point ID
section](../formats/sfmr-file-format.md#point-id-portable-3d-point-references)).
An edited reconstruction is a chain of bases with point edits between them,
so an id has to say which of those its index belongs to and has to keep
naming the same point as the chain grows. Two rules do that: the hash in an
id is always derived from the content of the thing that created the point,
a base or a point edit, never from a counter or a session; and an id is
minted against the **earliest** such thing the point's identity reaches. The
file form stays exactly what it is; a **session form** adds the node.

```
pt3d_{hash}_{index}            file form
pt3d_{hash}_{index}_n{node}    session form
```

| Part | Content | Example |
|------|---------|---------|
| `{hash}` | First 8 hex digits of the `content_xxh128` of the base the id is minted against, whether or not that base has been written to a file (below); or, for a point a point edit created, of that edit's content hash. | `a1b2c3d4` |
| `{index}` | The point's index in that base; or, for a point a point edit created, its index among the points that edit created. | `12345` |
| `{node}` | The node's session id (its `ReconId`), decimal. | `3` |

The session form stays in the `[a-zA-Z0-9_]` class, so it double-click
selects like the file form, and the file form is its prefix, so a session id
truncates to a file id by dropping the last field.

**Every base has a hash, and so does every point edit that creates one.** Both
are built and specified in
[`../core/reconstruction/edited-reconstruction.md`](../core/reconstruction/edited-reconstruction.md)
under "Hashes": a base's `content_xxh128` is computed from the value without
writing a file and equals what a save of it writes, and a point edit that
creates points is hashed over the base's hash plus the records it adds. So an
id minted against either names the same point after a save as before it, and
demo data is hashed the same way, which means `00000000` is not a state a node
can be in. What is left open here is the condition that equality rests on: the
hash covers the metadata section, so anything that stamps an operation, a tool
or a timestamp onto a base has to do so before the hash is taken, and where in
the save path that stamping belongs is a decision for the saving step.

### The version graph

The graph itself is built and standing:
[`../gui/edit-history.md`](../gui/edit-history.md) describes the per-step maps,
that they are kept for every version ever minted, and the forward and inverse
walks over one step. What is open here is the id minting and resolution built on
top of them.

A node's history is a chain of versions, and each step between two versions
carries a **point map**: a materialisation's row map from the old indexes to
the new, a point edit's record of which base index a modified point was
re-added under, and which indexes were deleted. A point's identity is
preserved across a modification: adding an observation to a track changes
the record, not which point it is, so the map says old index to new index
and the panel follows it. Deletion ends an identity, and materialisation
gives a modified point its base index back, shifts the points after a
deletion, and numbers the new ones.

The maps are kept for every version ever minted, including versions an undo
followed by a new edit discarded. They are tiny (the size of the edits, or of
a materialisation's survivors as one array), so the node holds the whole
graph of its session even when it has dropped the values. Every version's
serial is minted once and never reused.

**Resolving an id** finds the base or the point edit the hash names among
the node's versions, then walks the graph to the cursor's version. When the minting version is an
ancestor of the cursor's, the walk is forward through the maps in order. When
it is on a discarded branch, the walk is backward from it to the last common
ancestor, inverting each map (every map is a bijection on the points that
survive it, so this is well defined), then forward to the cursor. The walk
refuses, naming the version it stopped at, when the point was deleted along
either leg, or when it was added on the discarded leg and so never existed
on the surviving one. So an id copied at any point in the session, undone
past, edited over, and pasted back, still lands on its point if that point is
still there.

**Minting an id** walks the graph the other way. The id a panel shows for a
point is not its coordinate in the cursor's base but its coordinate in the
**earliest** base it existed in: the walk goes backward from the cursor
through the maps as far as the point's identity reaches, and mints against
the base, or the point edit, it stops at. For most points that is the file the node was loaded
from, so the id shown is the one that file's readers already use, and it
survives every edit of the session, every undo, and every save. The
earliest id is the one with the largest set of versions it resolves in, so
it is the one with the best chance of surviving whatever the user does next,
which is the point of copying an id.

**A point that is in no base yet.** A track added in the session exists only
in an overlay, so the earliest thing its identity reaches is the point edit
that created it, and it mints against that edit: `pt3d_{edit hash}_{k}`,
where `k` is its index among the points the edit created (zero for a single
new track; a split's second half is index one of its edit). Nothing about
that name comes from the session. Add a track, undo, and add a different
track, and the two ids differ, because the edits' contents differ, so the
first id resolves through the version graph to a miss on the discarded
branch rather than to the second track. Close without saving, reopen the
file tomorrow, and add a track: it collides with yesterday's id only if it
is the same observations at the same pixels, in which case it is the same
point. Two sessions adding different tracks on the same file mint different
ids.

The edit hash names a row of no file, which is what the out-of-range rule is
for, and a reader that meets it in a constraints file finds it among no
file's `content_xxh128`. It is found in **lineage**: when the overlay is
materialised the point gets a row in the new base and the row map records
the edit hash and index against that row, so the id keeps resolving, the
earliest rule keeps minting it afterwards, and the saved file's lineage
metadata carries the pair for any later session. An id that names an
addition never carried into a saved file is the one kind that dies with the
session, which is what never saving means.

The two walks are inverses, so an id minted this way and pasted back
resolves in one forward walk, and the same point always shows the same id
however many times it has been modified in between. Without the earliest
rule, two copies of one point's id taken before and after a materialisation
would differ, and the later one would die on an undo past that
materialisation.

**What the file form of a session id means.** For a point that existed
unchanged in a file the node was loaded from, the file form is exactly the
point's id in that file, so every id written down against that file (the
ground-truth table, a constraints file) keeps resolving through any number
of edits, and every id the panel shows for such a point is one the file's
readers can use. For a point added in the session the index is at or past
its base's count, so the file form is out of range in the file that base
was, or will be, saved as: a tool reading it detects that the id names a
point the file does not contain, rather than silently reading another point.

**What the suffix adds.** The node id makes the same file opened as two
nodes, with different edits, unambiguous, and lets Go to Point skip the hash
search. Go to Point accepts both forms; on the file form it searches every
node's version graph for the hash and prefers the selected node, as it does
today; a closed node is a miss, named as such.

**What the panels copy.** *Copy Point ID* copies the session form, minted by
the earliest rule. A constraints file wants file ids, and a tool consuming
one accepts a session id by taking its file-form prefix and applying the
out-of-range rule above.

**What a save does.** A save materialises if the overlay is not empty, which
is a new base with a new hash and a row map from the old indexes, and writes
the current base. Saving a base that is already materialised writes the
bytes its hash was computed from and mints nothing. Within the session the
version graph resolves any older id as above. Across sessions the graph is
gone unless the file carries it, so the saved file's metadata records its
lineage: the ancestor base's hash and the point map from it, the hashes of
the point edits whose points the materialisation carried in, chained back
through every base the session materialised, which is what lets an id from
last week's file land in this week's, and what lets the earliest rule keep
minting last week's ids after a save. That metadata entry is the one
addition this draft asks of the format spec, alongside a sentence in its
Point ID section that a reader may meet the session form and takes its
prefix.

Files into: `gui/goto-point.md` (both forms, the version graph, both walks),
the Point Track Detail's copy action, `gui/edit-history.md` (the graph as
part of the history), and the format spec's Point ID section (the one
sentence) and metadata (the lineage entry).

## Open questions

- The materialisation fraction, and whether it is measured in points, in
  observations, or in unshared bytes.
- Whether the version graph's maps are stored per version as they are minted
  or compacted into one map per base once a base is superseded; the walk is
  the same either way, and the first is simpler.
