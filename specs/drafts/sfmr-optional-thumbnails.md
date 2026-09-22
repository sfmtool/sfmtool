# Optional Thumbnails in `.sfmr`, and the `xform` Steps That Drop and Add Heavy Columns

**Status:** Draft. Decided: format version 11 makes the per-image thumbnail
column optional behind a `has_thumbnails` flag, and an empty
`workspace.absolute_path` means none was recorded; readers hold thumbnails as
optional in memory and never fill them in on load; the viewer builds display
thumbnails from the source photographs when a file has none; `sfm xform` gains
four ordered steps, `--drop-thumbnails`, `--drop-patch-bitmaps`,
`--add-thumbnails` and `--add-patch-bitmaps`, and a `--minimal` option that
drops the heavy columns and clears machine-local and incidental metadata,
lineage included. Every question the draft raised has been settled and is
folded into the text below; nothing remains open.

Amends [formats/sfmr-file-format.md](../formats/sfmr-file-format.md),
[cli/reconstruction/xform/xform-command.md](../cli/reconstruction/xform/xform-command.md),
[core/reconstruction/edited-reconstruction.md](../core/reconstruction/edited-reconstruction.md)
(the `ImageTable` shape),
[gui/multi-panel-image-browser.md](../gui/multi-panel-image-browser.md) and
[gui/camera-views.md](../gui/camera-views.md) (where the viewer's thumbnails
come from). Filing it also rewrites the Image Browser spec's "Thumbnail
loading" section, which says the browser loads thumbnails "from disk via the
`image` crate" when the code reads the embedded column; the rewrite describes
both sources as they will be, rather than leaving the old text beside the new.

## Purpose

A reconstruction file carries, beside the geometry, two columns of pictures: a
small preview of every photograph, so a viewer can show the images without
opening them, and a small texture for every surface patch, so a viewer can draw
the patches without re-rendering them. Both are conveniences. Neither says
anything about where the cameras are or where the points are, and together they
are nearly all of a file's bytes. This draft makes the image previews optional,
as the patch textures already are, adds four `sfm xform` steps that remove
either column from a file or put it back from the source photographs, and adds
`sfm xform --minimal`, which writes the smallest file that still holds the whole
reconstruction: no pictures, and none of the metadata that describes the
machine or the history it was made on.

The motivating case is a hand-edited ground truth for the
`seoul_bull_sculpture` test dataset, which is worth checking into the
repository once it is small:
`seoul_bull_ws/sfmr/ground_truth_candidate_tk104.sfmr`, a version 10
`embedded_patches` file of 17 images, 280 points and 1277 observations. It is
1.38 MB. The thumbnail entry
(`images/thumbnails_y_x_rgb.17.128.128.3.uint8.zst`) is 749 KB of that and the
patch bitmaps (`points3d/patch_bitmaps_y_x_rgba.280.24.24.4.uint8.zst`) are
591 KB; everything else is about 40 KB. The 17 photographs are already in the
repository under
[test-data/images/seoul_bull_sculpture/](../../test-data/images/seoul_bull_sculpture/),
so both columns can be rebuilt from what is checked in, and storing them a
second time costs 1.34 MB for nothing. With both dropped the file is about
39 KB, and `--minimal` (below) brings it to about 36 KB.

The thumbnails are also exactly recoverable. Each stored row of the tk104 file
equals, byte for byte, `cv2.resize(image, (128, 128), interpolation=INTER_AREA)`
of the matching photograph decoded with `IMREAD_IGNORE_ORIENTATION` and
converted to RGB (checked on all 17 images, maximum difference 0). That is the
resize the SIFT extractors perform
([extract_sfmtool.py](../../src/sfmtool/sift/extract_sfmtool.py)), and it is
why dropping the column loses nothing a reader cannot rebuild.

## Format: version 11

These are the changes to the `.sfmr` format spec. They are written as they will
read there, in that spec's terms.

### `images/metadata.json.zst`

```json
{
  "image_count": 17,
  "has_thumbnails": false,
  "thumbnail_size": null
}
```

- `has_thumbnails`: (version 11+) Whether
  `images/thumbnails_y_x_rgb.{N}.128.128.3.uint8.zst` is present. A version 11
  writer always writes the key, `true` or `false`.
- `thumbnail_size`: `128` when `has_thumbnails` is `true`, and `null` when it
  is `false`. As before, it restates the edge fixed by the entry name rather
  than parameterising it; a reader must not treat any other value as a
  description of the data.

A file of version 10 or earlier carries no `has_thumbnails` key and always
carries the entry, so it reads as `has_thumbnails: true`. A missing key means
the same thing in every version. This is the opposite default to the flags in
`points3d/metadata.json`, where a missing flag means absent, and it is chosen
for the same reason those were: a missing key must read as what every file
written before the key existed actually holds.

### `images/thumbnails_y_x_rgb.{N}.128.128.3.uint8.zst` (Optional from version 11)

The entry keeps its name, shape, dtype, layout and resize method. What changes
is its presence and what its presence asserts:

- **The column is whole or absent.** There is no per-image presence. A writer
  either stores a row for every image or stores no entry at all.
- **A present row is a downscale of the image it names.** Row `i` is the source
  photograph `images[i]` resized to 128 x 128 by area averaging, stretched to
  the square. A writer that cannot produce that row for some image (the
  photograph is missing, or cannot be decoded) writes no entry; it never
  stores a placeholder, because a flat or zero row in a present column would
  assert that the photograph looks like that.
- **Absence asserts nothing about the images.** A file without thumbnails is a
  complete reconstruction. A consumer that wants previews builds them from the
  photographs, which the workspace-relative `images/names` locate.

The existing Purpose line ("Enables instant thumbnail display in viewers
without requiring access to the workspace source images") and Source line
become a statement about files that carry the column; a file that omits it has
traded that convenience for size.

### Content hash

`images_xxh128` covers the thumbnail entry only when it is present, in its
existing lexicographic slot (after `sift_content_hashes`, before
`translations_xyz`). This is the rule `points3d_xxh128` already follows for
`patch_bitmaps_y_x_rgba`. The `images/metadata.json` bytes, which carry the
flag, are in the digest as they always were, so two files that differ only in
whether they carry thumbnails hash differently, and the difference is visible
in the metadata as well as in the entry list.

The thumbnails stay part of the reconstruction's identity (`content_xxh128`),
rather than moving to the unidentifying `derived/` section. `derived/` holds
values recomputable from the other sections of the file; thumbnails are
recomputable only from files outside it. Moving them would also not keep Point
IDs stable across `--drop-thumbnails`: an `xform` save rewrites `operation`,
`tool_options` and the workspace paths in `metadata.json`, which is hashed, so
its output has a new content hash regardless.

### An empty `workspace.absolute_path`

The format spec's path resolution gains one sentence, which the reader already
implements ([read.rs](../../crates/sfmtool-sfmr-format/src/read.rs), the
`absolute_path.is_empty()` guard in `resolve_workspace_dir`): **an empty
`absolute_path` means none was recorded**, and resolution skips that step. The
same holds for an empty `relative_path`. A file meant to travel between
machines, such as one checked into a repository, records `relative_path` and an
empty `absolute_path`, so that its hashed metadata holds nothing about the
machine it was written on. `sfm xform --minimal` writes files this way.

### Versioning

The new section of the format spec's Versioning and Migration, following the
shape of the version 5 to 6 and 6 to 7 sections, which introduced optional
columns the same way:

| Change | Detail |
|---|---|
| `images/metadata.json` `has_thumbnails` | New key, always written: whether `images/thumbnails_y_x_rgb` is present. Rides inside `images/metadata.json`, which is already hashed, so no hash slot changes. |
| `images/metadata.json` `thumbnail_size` | `null` when `has_thumbnails` is `false`. |
| `images/thumbnails_y_x_rgb` | **Optional.** Absent when `has_thumbnails` is `false`, and then not part of `images_xxh128`. |

Migration is mechanical and lossless in both directions. A version 10 file
reads as `has_thumbnails: true`; a version 11 file that carries thumbnails is
byte-identical in the images section to the version 10 file it came from apart
from the added `true` flag. A version 11 file without thumbnails has no version
10 equivalent until thumbnails are added back.

A conforming writer always writes the current version, as the writer has since
version 5 ([write.rs](../../crates/sfmtool-sfmr-format/src/write.rs) sets
`metadata.version = SFMR_FORMAT_VERSION` on every write). It does not choose 10
or 11 by whether thumbnails are present: every earlier optional column (normals
in 3, observation confidence in 6, constraints in 7) was introduced this way,
and a writer that picked its version by content would make "which version is
this file" a question about its columns. Nothing else rides along in version
11.

The Version History list gains: "**Version 11**: `images/thumbnails_y_x_rgb`
becomes optional, flagged by `has_thumbnails` in `images/metadata.json`; a
version 10 or earlier file reads as having thumbnails."

## In memory: core and bindings

Thumbnails are optional in memory for the same reason they are optional on
disk, and **a load never fills them in**. If a reader synthesised a column on
load, the next save of that value would write it, and a file dropped to 39 KB
would come back at 790 KB the first time anything touched it, with no step in
the chain that asked for it. Presence is a property of the value, carried from
load to save unchanged unless an operation changes it deliberately.

### `sfmtool-sfmr-format`

[`SfmrData::thumbnails_y_x_rgb`](../../crates/sfmtool-sfmr-format/src/types.rs)
becomes `Option<Array4<u8>>`, like `patch_bitmaps_y_x_rgba`. The reader reads
the entry when the flag says it is present; the writer validates the shape
only when it is `Some`, writes the flag, and skips the entry and its hash
contribution when it is `None`; the verifier
([verify.rs](../../crates/sfmtool-sfmr-format/src/verify.rs)) reads the entry
only when the flag is set. `content_hash_of` shares the writer's hashing path,
so the in-memory hash of a value without thumbnails agrees with what its save
stores. `THUMBNAIL_SIZE` is unchanged.

### `sfmtool-core`

[`ImageTable::thumbnails_y_x_rgb`](../../crates/sfmtool-core/src/reconstruction/data/image_table.rs)
becomes `Option<Arc<Array4<u8>>>`, the shape `PointSet::patch_bitmaps_y_x_rgba`
already has. The `Arc` stays for the reason its doc gives: two values that
agree on their thumbnails share one allocation.

| Site | Today | Without thumbnails |
|---|---|---|
| [data/conversion.rs](../../crates/sfmtool-core/src/reconstruction/data/conversion.rs) | wraps the loaded array; clones it back out on save | maps the `Option` both ways |
| [edit.rs](../../crates/sfmtool-core/src/reconstruction/edit.rs) `subset_by_image_indices` | builds a new column of the kept rows | builds one when the input has one, `None` otherwise |
| [edit.rs](../../crates/sfmtool-core/src/reconstruction/edit.rs), [edited.rs](../../crates/sfmtool-core/src/reconstruction/edited.rs), [prune_covered.rs](../../crates/sfmtool-core/src/reconstruction/prune_covered.rs) | share the input's `Arc` (point-only edits, materialisation) | share the input's `Option<Arc>`; nothing to change beyond the type |
| [data/demo.rs](../../crates/sfmtool-core/src/reconstruction/data/demo.rs) | a zero column | `None`: demo data has no photographs, and a zero column is the placeholder rows the format forbids |
| [lib.rs](../../crates/sfmtool-core/src/lib.rs) | re-exports `THUMBNAIL_SIZE`; asserts it equals the `.sift` edge | unchanged |

The ImageTable section of
[edited-reconstruction.md](../core/reconstruction/edited-reconstruction.md)
shows the field's type and changes with it.

### `sfmtool-py`

The Python surface follows `patch_bitmaps`, which is already optional:

- `SfmrReconstruction.thumbnails_y_x_rgb`
  ([sfmr_reconstruction.rs](../../crates/sfmtool-py/src/reconstruction/sfmr_reconstruction.rs))
  returns the read-only zero-copy view, or `None` when the value has no
  thumbnails.
- `clone_with_changes(thumbnails_y_x_rgb=None)`
  ([clone.rs](../../crates/sfmtool-py/src/reconstruction/clone.rs)) drops the
  column; an `(N, 128, 128, 3)` `uint8` array sets it, with the existing shape
  check.
- `read_sfmr` ([io/sfmr.rs](../../crates/sfmtool-py/src/io/sfmr.rs)) returns
  `None` under `thumbnails_y_x_rgb` for a file without them, and `write_sfmr`
  takes the key as optional (`get_optional_item`), as it already takes
  `feature_tool_hashes`.
- `sfmtool.THUMBNAIL_SIZE` is unchanged.

### Python consumers

Nothing under [src/sfmtool/](../../src/sfmtool/) needs thumbnails to do its
work. The survey:

| Consumer | Today | Without thumbnails |
|---|---|---|
| [merge/reconstructions.py](../../src/sfmtool/merge/reconstructions.py) | stacks each input's row for every merged image | output has thumbnails only when every input does; otherwise `None` and one printed line saying so. Never synthesises. |
| [merge/pose_refinement.py](../../src/sfmtool/merge/pose_refinement.py) | passes the merged column through | passes `None` through |
| [colmap/io.py](../../src/sfmtool/colmap/io.py) | producer: copies each image's `.sift` thumbnail into a new reconstruction (solve, `from-colmap-bin`) | unchanged; a producer that has the `.sift` files writes the column |
| [_undistort_images.py](../../src/sfmtool/_undistort_images.py) | producer: thumbnails of the undistorted images | unchanged |
| [sift/file.py](../../src/sfmtool/sift/file.py), the three extractors | `.sift` thumbnails | unchanged; the `.sift` format keeps its thumbnail mandatory |
| [scripts/measure_edit_costs.py](../../scripts/measure_edit_costs.py) | sums the heavy columns | counts an absent column as zero bytes |

Everything else that reads a reconstruction (every `xform` step, `inspect`,
`analyze`, `compare`, the COLMAP and Nerfstudio exports, `render-patches`,
`panorama`) never reads the column, so it runs unchanged on a file without it.
No consumer refuses a file without thumbnails and none synthesises them; the
one place thumbnails are made from photographs for an existing reconstruction
is `sfm xform --add-thumbnails`, asked for by name.

`sfm inspect`'s reconstruction summary
([analyze/summary.py](../../src/sfmtool/analyze/summary.py)), which prints
neither today, gains two lines, `Thumbnails: yes` or `no` and
`Patch bitmaps: 24x24` (the stored resolution) or `no`, since a file's size and
whether the viewer will need the photographs both turn on them. It prints an
empty absolute workspace path as `(none recorded)` rather than a blank.

## The viewer

The viewer shows a thumbnail on every frustum's far plane, in every Image
Browser cell, beside every Track View row, and in the Image Browser's colour
barcode. When a file carries thumbnails it keeps drawing those. When it does
not, **the viewer builds display thumbnails from the source photographs** and
draws those instead.

**Display thumbnails are the node's, not the value's.** They live on the scene
node beside its history, the way its open SIFT index does
([sift_index.rs](../../crates/sfm-explorer/src/sift_index.rs)), and never enter
the `ImageTable`. So nothing the viewer synthesises can reach a save: a file
opened without thumbnails is saved without them, whatever was drawn. For a file
that has them, the display column is the image table's own `Arc`, shared
rather than copied.

**They are keyed by image name.** Delete Image renumbers the image table, and
Undo renumbers it back, so a column addressed by index would need rebuilding on
every such step. Addressed by the workspace-relative name the table carries,
one column built when the node opens serves every version of the node.

**Each row is the format's resize.** The photograph is read from
`workspace_dir.join(name)`, the path Image Detail already reads full-resolution
images from ([state.rs](../../crates/sfm-explorer/src/state.rs),
`decode_full_res`), without applying EXIF orientation (the `image` crate's
default, and the extractors' `IMREAD_IGNORE_ORIENTATION`), and resized to
128 x 128 by area averaging. The area resize is a small function in
`sfmtool-core` shared by any Rust caller; the viewer uses it for display only,
so it matches OpenCV's `INTER_AREA` in method rather than bit for bit.

**A photograph that cannot be read gets a flat placeholder.** The row is a
neutral mid-grey, and the image's name is logged once. If no photograph can be
read at all (the workspace did not resolve, or it holds no images), the node
has no display column, the frustums draw their outlines without image quads
(the existing no-atlas path in
[frustums.rs](../../crates/sfm-explorer/src/scene_renderer/upload/frustums.rs)),
and the Image Browser and Track View draw the placeholder rectangle they
already draw for a texture not yet loaded.

**Synthesis runs off the GUI thread and fills in progressively.** It is not a
document operation: it changes no value, costs no version, and must not occupy
the one background-task slot that edits and adjustments use
([gui/background-tasks.md](../gui/background-tasks.md)), or a large capture
would lock out editing until it finished. Decoding and resizing costs about
1.3 ms per `seoul_bull_sculpture` image (270 x 480) and 13 ms per
`dino_dog_toy` image (2040 x 1536) with OpenCV on one thread, so the 17-image
set is instant and a capture of thousands of full-size photographs takes
minutes. So synthesis has a worker pool of its own, fills rows in Image
Browser order starting from the cells on screen, has no cap on the image count,
and records one line in the Action Log when it finishes. Each finished row is
visible on the next frame; until then its cell shows the placeholder.

Per consumer:

| Site | Change |
|---|---|
| [scene_renderer/upload/thumbnails.rs](../../crates/sfm-explorer/src/scene_renderer/upload/thumbnails.rs) | builds the atlas from the node's display column in the value's image order; the reuse check keys on the display column and the image list rather than on `image_table.thumbnails_y_x_rgb` by pointer; re-uploads rows as synthesis completes them |
| [scene_renderer/recon.rs](../../crates/sfm-explorer/src/scene_renderer/recon.rs) | `uploaded_thumbnails` holds the display column it was built from |
| [scene_renderer/upload/frustums.rs](../../crates/sfm-explorer/src/scene_renderer/upload/frustums.rs), [pipelines/image_quad.rs](../../crates/sfm-explorer/src/scene_renderer/pipelines/image_quad.rs) | unchanged; they already key on whether the atlas exists |
| [image_browser.rs](../../crates/sfm-explorer/src/image_browser.rs) | cells and the barcode read the display column; the barcode is built once every row is final |
| [track_view/view/table.rs](../../crates/sfm-explorer/src/track_view/view/table.rs) and the texture caches in [track_view/view/mod.rs](../../crates/sfm-explorer/src/track_view/view/mod.rs) | row thumbnails read the display column; a row that is still a placeholder is not cached as final |
| [texture.rs](../../crates/sfm-explorer/src/texture.rs) | unchanged: `thumbnail_color_image` takes a view of whatever column it is handed |
| [document.rs](../../crates/sfm-explorer/src/document.rs) `value_bytes` | counts the image table's column when present, as it does the bitmaps; a node's synthesised display column is counted once per node, not per version |
| [app.rs](../../crates/sfm-explorer/src/app.rs) | the `thumbnails` upload phase runs as now, and again when synthesis has new rows |
| [scene.rs](../../crates/sfm-explorer/src/scene.rs) | unchanged (the tint and highlight colours only name thumbnails) |

A file without patch bitmaps already loads; the viewer draws such patches as
[patch-rendering-flat-shaded-amendment.md](patch-rendering-flat-shaded-amendment.md)
proposes, and nothing here changes that.

## `sfm xform` steps

Four steps, run in command-line order like every other step, each a
`Transform` returning a new reconstruction, and one shorthand over two of them,
[`--minimal`](#--minimal):

```bash
sfm xform in.sfmr out.sfmr --drop-thumbnails
sfm xform in.sfmr out.sfmr --drop-patch-bitmaps
sfm xform in.sfmr out.sfmr --add-thumbnails
sfm xform in.sfmr out.sfmr --add-patch-bitmaps [resolution=<R>,sampler=<S>]
```

An `--add-*` step is a no-op, with one printed line, on a reconstruction that
already carries the column. To re-render at a different resolution, drop first:
`--drop-patch-bitmaps --add-patch-bitmaps resolution=32`.

### Names

The glossary ([GLOSSARY.md](../GLOSSARY.md)) has no entry on these verbs, so
the choice is argued from the `xform` vocabulary as it stands:

- `--remove-*` (`--remove-short-tracks`, `--remove-isolated`, ...) and
  `--filter-by-*` remove **points**, with their observations, and renumber the
  rest. A `--remove-thumbnails` would read as the same kind of operation and is
  not: it removes no row of anything.
- `--include-*` and `--exclude-*` select **images**.
- **drop** is already the word this code uses for discarding an optional
  column while keeping every row: `--localize-keypoints` "drops" stale patch
  bitmaps (its spec and help text), and `clone_with_changes` documents `None`
  as the way "to drop" `normal_confidence`.
- **add** is drop's inverse and says only that the column appears. The
  alternatives each say something false or ambiguous: `embed` collides with the
  `embedded_patches` feature source and `sfm embed-patches`, and `render`
  describes how bitmaps are made but not thumbnails, which are resized.

Filing this draft adds a glossary entry: *drop* discards an optional column and
keeps every row, *remove* deletes points (rows), *add* fills an absent optional
column from the source data, each in the `xform` step vocabulary and the
binding keywords behind it.

### `--drop-thumbnails`

`recon.clone_with_changes(thumbnails_y_x_rgb=None)`. Reads no files, keeps
every row of everything. Valid on both feature sources.

### `--drop-patch-bitmaps`

`recon.clone_with_changes(patch_bitmaps=None)`. The patch frames
(`patch_u_halfvec_xyz`, `patch_v_halfvec_xyz`) and the normals stay, so the
patches keep their geometry and a later `--add-patch-bitmaps`,
`--refine-keypoints` or `--refine-normals` can render onto them. Valid on both
feature sources; a no-op, with one printed line, on a reconstruction without
bitmaps.

### `--add-thumbnails`

Builds the column from the **source photographs**: for each image,
`workspace_dir / name` decoded with `cv2.IMREAD_COLOR | IMREAD_IGNORE_ORIENTATION`,
resized to 128 x 128 with `INTER_AREA`, converted to RGB. That is the
extractors' own code path, so the result is byte-identical to the `.sift`
thumbnail an extractor would have written for the same photograph, and to the
row a file originally carried (shown on tk104 above). The decode shares its
orientation flag with the extractor; note that
[_workspace_image.py](../../src/sfmtool/_workspace_image.py)
`read_workspace_image` does not pass `IMREAD_IGNORE_ORIENTATION` today, so this
step calls the extractor's decode rather than that helper.

In an `embedded_patches` file each photograph is first checked against the
image's stored `image_file_hashes` entry, the hash that exists to say "this is
still the photograph the reconstruction was built from". A mismatch or a
missing photograph fails the step, naming every such image, because the column
is whole or absent and the format forbids a placeholder row.

**The photograph is preferred over the image's `.sift` copy.** A thumbnail is
defined as a downscale of the photograph, so the photograph is the source and
the `.sift` is a cache of it. The photograph is reachable from both feature
sources, while an `embedded_patches` file records no `.sift` link at all
(no `sift_content_hashes`, no `feature_tool_hashes`), so a `.sift` copy there
would be found by guessing a path from `feature_prefix_dir` and trusted
unverified. And since both paths run the same resize on the same decode, the
photograph gives up nothing in fidelity. The `.sift` copy is the fallback when a
photograph is missing and the image's `.sift` can be verified: in a
`sift_files` file, by its content hash against `sift_content_hashes`; in an
`embedded_patches` file, by its recorded `image_file_xxh128` against
`image_file_hashes`. The reverse order would be faster on a large capture (one
small archive entry against a full decode), but it would trust a cache over its
source, and in an `embedded_patches` file it has no recorded link to find that
cache by.

### `--add-patch-bitmaps [resolution=<R>,sampler=<S>]`

Renders an RGBA bitmap for every point that has a patch frame, **at the
patch's stored frame and each observation's stored keypoint, moving nothing**:
positions, normals, frames, keypoints and tracks come out exactly as they went
in. It reads the source photographs, and requires an `embedded_patches`
reconstruction with patch frames, like the other patch steps
(`required_feature_source = "embedded_patches"`). A point with fewer than two
observations that render in frame gets a zero row, as `--refine-keypoints`
gives it.

`resolution` defaults to 24, the default of
`KeypointSubpixelParams::resolution`
([keypoint_subpixel/params.rs](../../crates/sfmtool-core/src/patch/keypoint_subpixel/params.rs))
and of the `resolution` key of `--refine-keypoints` and `--refine-normals`. It
cannot default to the file's `patch_bitmap_resolution`: the step does anything
only when the file has no bitmaps, and then that field is `null`. `sampler`
takes the values and default of the `sampler` key of `--refine-keypoints`.
Those two are the only keys: the other sub-pixel parameters tune a solve this
step does not run.

**Where bitmaps are rendered today, and the render this step factors out.**
Three paths render patch bitmaps:

- `--refine-keypoints` ([_refine_keypoints.py](../../src/sfmtool/xform/_refine_keypoints.py))
  calls `PatchCloud.refine_keypoints(..., render_bitmaps=True)`; the kernel
  ([keypoint_subpixel.rs](../../crates/sfmtool-core/src/patch/keypoint_subpixel.rs),
  `render_representative`) fuses each point's representative at the final
  per-view keypoints with the final IRLS view weights.
- `--refine-normals` ([_refine_normals.py](../../src/sfmtool/xform/_refine_normals.py))
  calls `PatchCloud.refine_normals(..., render_bitmaps=True)`; the kernel
  ([normal_refine/mod.rs](../../crates/sfmtool-core/src/patch/normal_refine/mod.rs))
  fuses the winning normal's view stack with its consensus weights and the
  obliquity prior, over the views that normal kept.
- The bench commit ([bench/fit.rs](../../crates/sfmtool-core/src/bench/fit.rs),
  `fuse_bitmap`) already does exactly what this step needs for one track: it
  runs the sub-pixel kernel with `max_gn_steps: 0`, `max_outer_sweeps: 1` and
  `render_bitmaps: true`, so that "nothing moves ... this pass is the fuse", at
  the reconstruction's own bitmap resolution when it stores one.

`--to-embedded-patches` renders no bitmaps; it builds frames from the mean
viewing direction and stops there.

The factoring lifts the bench's zero-step fuse into the patch module as the one
place a representative is rendered for a patch whose placement and keypoints
are settled:

```rust
// crates/sfmtool-core/src/patch/keypoint_subpixel.rs
/// Fuse `patch`'s RGBA representative from `views` at `keypoints`, moving
/// nothing. `None` when fewer than two views render in frame.
pub fn fuse_patch_bitmap(
    patch: &OrientedPatch,
    images: &[ProjectedImage<'_>],
    view_set: &[u32],
    keypoints: &[[f64; 2]],
    params: &KeypointSubpixelParams,
) -> Option<Vec<u8>>;
```

`bench::fit::fuse_bitmap` calls it for one track. `PatchCloud` gains a
whole-cloud form, parallel over points, bound as
`PatchCloud.render_bitmaps(recon, images, resolution=24, sampler=..., progress=None)`
and returning the `(P, R, R, 4)` array `clone_with_changes(patch_bitmaps=...)`
takes. `--add-patch-bitmaps` is that call plus the write-back.

Its bitmaps therefore equal what `--refine-keypoints` renders for a patch whose
keypoints it did not move, and what the bench commits. They are not
byte-identical to `--refine-normals` bitmaps, which fuse over a different view
subset with an obliquity weight; a file whose bitmaps came from
`--refine-normals` does not round-trip bit-exactly through
`--drop-patch-bitmaps --add-patch-bitmaps`.

### What the steps do not touch

None of the four moves a point, renumbers a row, or changes a keypoint, so the
input's `lineage` (which maps ancestor points onto this file's rows) stays
valid across them. They keep `xform`'s ordinary lineage behaviour, which is to
write the input's list into the output unchanged and add no entry of its own;
nothing in the tree writes a new lineage entry today (the viewer stopped, for
the reasons in [gui/saving.md](../gui/saving.md)). Passing the list through is
right for these steps and wrong for any step that renumbers points, which is a
defect of the point filters rather than of anything here. The `derived/`
section is recomputed by the save as always and is unchanged by any of them.

### `--minimal`

`--minimal` writes the smallest file that still holds the whole
reconstruction. It is **a shorthand for `--drop-patch-bitmaps
--drop-thumbnails`, plus a clearing of the metadata that is machine-local or
incidental**, and it is what a file meant for a repository is written with:

```bash
sfm xform in.sfmr out.sfmr --minimal
```

The output is `xform`'s second positional argument, and the steps follow the
positionals, so the motivating file is made with
`sfm xform ground_truth_candidate_tk104.sfmr <out>.sfmr --minimal`.

It has two parts, which take effect at different times:

- **The column part is an ordered step.** At its position in the chain it does
  what `--drop-patch-bitmaps --drop-thumbnails` would do there. A later step
  sees a reconstruction without either column, so `--minimal --add-thumbnails`
  writes a file with thumbnails and minimal metadata, and a patch step after
  it renders bitmaps as it would after the two drops. An `--add-*` step that
  restores part of the shorthand prints one line saying so, so the
  combination reads as intended rather than as a mistake.
- **The metadata part is a property of the save.** The save rewrites
  `operation`, `tool`, the counts and both workspace paths after every step has
  run, so clearing metadata at the step's position would be undone. `--minimal`
  anywhere in the chain therefore marks the output, and the save writes it
  minimal. Giving it twice is the same as giving it once.

The heavy columns go because they are the size: on tk104 they are 1.34 MB of
1.38 MB. Thumbnails are in the shorthand as well as bitmaps because the purpose
is the smallest file: the thumbnails are the larger of the two (749 KB of
tk104's 1.38 MB), both are rebuilt exactly from the photographs by the
`--add-*` steps, and a `--minimal` that kept one would need a second flag beside
it on every use it was made for. A caller that wants the thumbnails kept writes
`--minimal --add-thumbnails`.

**`--minimal` drops `lineage` entirely.** The output is a new file with no
ancestry: it carries none of the input's entries and gains none for the input.
Point IDs minted against the input or any earlier version therefore do not
resolve in the minimal file, and that is intended. A minimal file is a root,
typically published somewhere its ancestors are not, so an entry could never
lead a reader to an ancestor file; and an inherited list is only as good as the
chain that wrote it. tk104's does not describe tk104: its `metadata.json`
carries 45 entries (about 90 KB of JSON before compression), and not one maps
onto its 280 rows. The newest, for example, is a `monotone` map with
`source_rows` 811, one `deleted` and no `created`, which pairs 810 ancestor
points with this file's rows.

**What it clears and what it keeps**, surveyed on every field tk104 carries:

| Field (tk104's value) | `--minimal` | Why |
|---|---|---|
| `metadata.json` `workspace.absolute_path` (`C:\Dev\prod\sfmtool\seoul_bull_ws`) | **cleared** to `""` | Names one machine's filesystem, and sits in `metadata_xxh128`, so the same reconstruction written on two machines would hash differently. An empty value means none was recorded. |
| `workspace.relative_path` (`..`) | recomputed by the save, from the output's directory to the workspace, as every `xform` save does | It is how a reader finds the workspace from where the file is. |
| `workspace.contents` (`feature_tool`, `feature_type`, `feature_options`, `feature_prefix_dir`) | kept | Not machine-local: it states which extractor and settings the reconstruction's features came from, and `feature_prefix_dir` is how the SIFT index build ([sift_index.rs](../../crates/sfm-explorer/src/sift_index.rs), `workspace_metadata`) and every `.sift`-reading step locate the features. |
| `lineage` (45 entries) | **dropped**, and no entry added | Above. |
| `tool_options` (`{"algorithm": "global"}`, inherited from the solve that began the chain) | **replaced** by `{"transforms": [...]}`, this invocation's own step list | The save *merges* its `transforms` record into the input's options, so a file accumulates the options of every operation behind it. The inherited keys describe an ancestor, which a root does not have; this invocation's list is how the file was made, and is kept. An empty object would make two minimal files of one input, made by different chains, hash alike, saving a few dozen bytes and saying less about the file. |
| `operation`, `tool`, `tool_version` (`edit`, `sfm-explorer`, `0.2.0`) | rewritten by the save as for any `xform` (`xform`, `sfmtool`, its version) | They describe the operation that wrote this file. |
| `version`, `feature_source`, the counts | kept, recomputed by the save | Format facts. |
| `world_space_unit` (absent on tk104) | kept when present | It states what the coordinates mean. |
| `written.json` `timestamp` | written by the save as always | Outside every section hash, so it adds nothing to the identity; a version 8+ writer always writes the entry. |
| `derived/` (depth statistics and histograms) | recomputed by the save, kept | Recomputable from the poses, positions and tracks, and outside `content_xxh128` already; 1.8 KB. |
| `content_hash.json` | recomputed | The digest. |
| `images/image_file_hashes` | kept | The identity of each photograph, which is what lets `--add-thumbnails` (and any reader) check that the photographs found are the ones reconstructed. 282 bytes. |
| `images/thumbnails_y_x_rgb` | **dropped** | The shorthand. |
| `points3d/patch_bitmaps_y_x_rgba` | **dropped** | The shorthand. |
| `points3d/patch_u_halfvec_xyz`, `patch_v_halfvec_xyz`, `normals_xyz` | kept | Geometry: the patches keep their placement, so bitmaps can be rendered back onto them. |
| positions, colours, reprojection errors, tracks, keypoints, poses, cameras, image names | kept | The reconstruction itself. The colours are picture-derived too, but 850 bytes and read by every point display. |

Optional columns tk104 does not carry follow the same line: `normal_confidence`,
`observation_confidence` and the point-constraint triple are measurements or
solve state and are kept when present. Nothing beyond the table is cleared.
Colours and reprojection errors could be recomputed, but each is under 1 KB on
tk104 and is either a statement about the reconstruction or read by every
consumer that displays points; `--minimal` removes the columns that are the
size and the metadata that describes a machine or a history, not every byte
that could be recomputed.

## Testing

- **Format crate** ([tests.rs](../../crates/sfmtool-sfmr-format/src/tests.rs)):
  a value without thumbnails round-trips, writes `has_thumbnails: false` and
  `thumbnail_size: null`, stores no thumbnail entry, and verifies; its
  `images_xxh128` equals the digest over its entries; `content_hash_of` equals
  what the write stores; a version 10 fixture without the key reads as having
  thumbnails; a writer given a wrong-shaped column still refuses.
- **Core**: `subset_by_image_indices` on a value without thumbnails yields one
  without; with thumbnails, the kept rows in order.
- **Bindings** (`tests/rust_bindings/test_sfmr_io_rust_bindings.py`): the
  getter returns `None`; `clone_with_changes(thumbnails_y_x_rgb=None)` drops;
  `read_sfmr` and `write_sfmr` accept the absent column.
- **xform** (`tests/xform/`): `--drop-thumbnails --add-thumbnails` restores a
  column byte-identical to the input's on the `seoul_bull_sculpture` fixture;
  `--drop-patch-bitmaps` keeps frames and normals; `--add-patch-bitmaps` changes
  nothing but the bitmap column and matches what the zero-step fuse renders;
  both `--add-*` steps are no-ops on a column already present; a missing
  photograph fails `--add-thumbnails` naming the image.
- **`--minimal`**: the output has neither column, an empty `absolute_path`,
  a `relative_path` from its own directory, no `lineage`, and `tool_options` holding only
  its own `transforms`; geometry, tracks, keypoints and `image_file_hashes` are
  equal to the input's; `--minimal --add-thumbnails` has thumbnails and minimal
  metadata; the same input written by `--minimal` twice, to the same place
  relative to its workspace, has the same `content_xxh128` whatever machine
  it was written on, by one sfmtool version.
- **Viewer** lib tests: a node opened from a file without thumbnails has a
  display column and an image table with none, and saves without them; a node
  whose photographs are absent opens and draws outlines.

## Non-goals

- The `.sift` format is unchanged: its thumbnail stays mandatory.
- No per-image thumbnail presence. The column is whole or absent.
- No thumbnail edge other than 128; the entry name still fixes it.
- The viewer has no command that writes its display thumbnails into a value;
  `sfm xform --add-thumbnails` is how a file gains them.
