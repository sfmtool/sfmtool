# `sfm web-export` Command

`sfm web-export` turns an `.sfmr` reconstruction into a directory that a web
browser shows as an interactive 3D view: the points, their oriented patches
textured with the patch bitmaps, and the cameras as frustums with each image's
thumbnail on the far surface, the way SfM Explorer draws them. The directory is
plain static files with no server-side code, so it can be served by any static
host, embedded in another page with an `<iframe>`, or published as the
supporting files of a claude.ai artifact. That last use is the one it is shaped
for: a report about a reconstruction can show the reconstruction itself, and a
reader can turn it with a mouse, or with touch on a phone.

It is a viewer and nothing more. Nothing in the page edits the reconstruction,
and the page reads no `.sfmr`, no `.sift` and no photographs: everything it
draws is baked into the directory by the command, including every camera-model
computation, so the browser draws only triangles, lines and points.

## Command Syntax

```bash
sfm web-export RECON.sfmr -o OUT_DIR [OPTIONS...]
```

Registered under the Visualization category, beside `sfm render-patches` and
`sfm panorama`.

| Option | Default | Meaning |
|--------|---------|---------|
| `-o, --output DIR` | required | Directory to write. Created if missing; refused if it exists and is not empty, unless `--overwrite`. |
| `--overwrite` | off | Remove the files this command writes (`scene.json`, `index.html`, `web-export.js`, `patches-<n>.jpg`, `thumbs-<n>.jpg`) from an existing directory, then write. Other files there are kept. |
| `--no-patches` | off | Leave out the patch atlas; points are drawn as round splats in their stored colour. |
| `--no-thumbnails` | off | Leave out the thumbnail atlas; frustums are drawn as wireframes only. |
| `--patch-size PX` | the file's bitmap size | Resample patch tiles to this many texels a side (for example 12) by area averaging, to shrink the atlas. |
| `--jpeg-quality Q` | 85 | JPEG quality for both atlases, 1 to 100. |
| `--max-points N` | none | Keep the N points with the most observations, ties going to the lower index; the rest are left out. |
| `--start-image NAME` | none | Open the page looking through this image's camera instead of framing the whole scene. The name is the image name the reconstruction stores; an unknown name is refused before anything is written. |
| `--single-file` | off | Write one self-contained `index.html` with the data, the atlases and the viewer inline, refused if it would pass 16 MB. |

The help says that the thumbnails of the photographs are in the output, so
sharing the directory or the artifact shares them.

**Thumbnails come from the file** when it carries the column. When it does not,
each image's row is its `.sift` thumbnail, or its photograph decoded and resized,
or a flat grey placeholder, in that order: the rows SfM Explorer's open builds
for its display thumbnails ([GLOSSARY](../../GLOSSARY.md) "display thumbnails"),
made by the same function. When no image yields a row, the thumbnail atlas is
left out.

**Patch bitmaps come from the file** when it carries them. A file with patch
frames and inline keypoints but no bitmaps has them rendered from its
photographs by the fuse SfM Explorer's open runs for its display patch bitmaps;
when no photograph can be read, the points are drawn as splats. A point with no
frame, an all-zero bitmap row or a place at infinity is drawn as a splat either
way.

The command prints the points written, the patches and where their bitmaps came
from, where the thumbnails came from, the GPU memory the atlases take once
decoded, and each file with its size. When the decoded atlases pass 256 MB it
warns, naming `--patch-size 12` (which quarters the patch atlas) and
`--max-points`, since a phone may refuse that much texture memory or reload the
tab.

## What it writes

```
OUT_DIR/
  index.html        the viewer page; open it from a server or point an iframe at it
  web-export.js     the viewer, an ES module: mount(element, scene, options)
  scene.json        what the scene holds, the per-point arrays included
  patches-0.jpg …   patch colour atlas pages, 4096 by 4096 at most
  thumbs-0.jpg …    thumbnail atlas pages, 4096 by 4096 at most
```

`index.html` and `web-export.js` are the same bytes for every scene of one
sfmtool version. They live in the Python package, under
[src/sfmtool/web_export/](../../../src/sfmtool/web_export/), and the command
copies them in; they are package data rather than strings compiled into the
Rust crate, so they are edited without a rebuild. Everything scene-specific is
in the other files.

A page's files are all web file types, because a claude.ai artifact serves only
a fixed list of them (pages, scripts, JSON and text, images, fonts, media, PDF)
and refuses to publish a raw `.bin`. That is why the point arrays travel inside
`scene.json` as base64 rather than as a binary file of their own. Base64 costs a
third more bytes before compression and little after it, and saves a fetch.

### `scene.json`

| Key | Holds |
|-----|-------|
| `format` | `1`, the layout described here |
| `generator` | `sfmtool <version>` |
| `source` | `name`, the `.sfmr` file name, and `content_xxh128`, its content hash |
| `center` | the scene centre in the reconstruction's frame, `f64`: the per-axis median of the finite points |
| `radius` | the 80th-percentile distance of the finite points from `center` |
| `point_size` | the splat radius, SfM Explorer's automatic point size (the trimmed median nearest-neighbour distance) |
| `length_scale` | SfM Explorer's length scale: ten splat radii, or the camera spacing when that is smaller |
| `frustum_length` | half the length scale, SfM Explorer's default frustum size |
| `view` | the initial view: `eye`, `target`, vertical `fov` in degrees, and `start_image` (a name or `null`) |
| `points` | `count`, `at_infinity`, `left_out` (by `--max-points`), `with_patch`, and `arrays`, the layout of `points_b64` |
| `points_b64` | the point arrays, one base64 byte block |
| `cameras` | per image: `name`, `center`, `rotation_wxyz` (world from camera), `grid` (`[first vertex, vertices per edge]` in `frustums_b64`) and `thumb` (`[page, cell]` in the thumbnail atlas, absent without one) |
| `frustum_arrays`, `frustums_b64` | the cameras' far-surface grid vertices, one base64 byte block |
| `atlases` | `patches` and `thumbnails`, each `null` or `pages` (`file`, `width`, `height`), `size` (tile texels), `tile` (with border), `border` (1), `cols` and `per_page` |

Every coordinate is in the reconstruction's frame (Z up, a camera looking down
its own −Z) with `center` subtracted, so a georeferenced scene far from the
origin keeps its precision in `f32`. A point at infinity keeps its unit
direction as it is. The framing view looks at the centre from 30 degrees above
the side the cameras are on, far enough back for the larger of `radius` and the
90th-percentile camera distance to fill a 50-degree view. A `--start-image` view
puts the eye at that camera's centre and the target on its optical axis, as far
out as the scene centre lies along it, with the camera's vertical field of view
held to 20 to 100 degrees.

### The byte blocks

Each block is little-endian arrays one after another, each starting on a 4-byte
boundary. Its `arrays` entry gives each array's `offset` in bytes, `type`
(`f32`, `u32`, `u16`, `u8`), `components` and `count`.

| `points` array | Type | Per point |
|----------------|------|-----------|
| `position` | `f32 × 3` | place relative to `center`, or unit direction when at infinity |
| `patch_u`, `patch_v` | `f32 × 3` each | half-vectors of the patch frame, zero when the point is drawn as a splat; present only with a patch atlas |
| `patch_cell` | `u32` | atlas page in the high byte, cell in the low 24 bits; all ones for a splat; present only with a patch atlas |
| `color_w` | `u8 × 4` | stored colour, then 1 for a place or 0 for a direction |
| `observations` | `u16` | track length, clamped to 65,535; shown on hover |
| `source_index` | `u32` | the point's index in the `.sfmr`; present only when `--max-points` left points out |

`frustum_arrays` has one array, `vertex`, `f32 × 3`. A camera's grid is
`vertices per edge` squared vertices, row major from the top-left pixel corner
to the bottom-right one. A pinhole camera without distortion has two per edge,
its far-plane corners. Any other camera has nine per edge, each vertex its
pixel's ray at `frustum_length`: on the far plane for a perspective lens, and on
the sphere of that radius for a fisheye, so the image surface bends the way the
lens does. This is SfM Explorer's frustum grid, from the same function.

### Atlases

Patch tiles are packed in point order and thumbnails in image order into a grid
of cells, each tile ringed by a one-texel border copied from its own edge, so
bilinear sampling at a tile's edge never reads its neighbour. A set that fits one
page is laid out close to square, so a small scene writes a small page; a larger
one fills pages of `4096 / tile` cells a side, and the last page is cut to the
rows it uses. Every page has the same `cols`, so a cell index finds its texel on
any page. Pages are baseline JPEG. The patch bitmaps' alpha channel is left out:
it is per-pixel cross-view confidence, not transparency, and SfM Explorer's
default edge cutoff of 0 draws every texel opaque anyway.

Pages are at most 4096 by 4096, the texture size every current phone's WebGL2
supports. A decoded page takes 64 MB of GPU memory whatever its JPEG size, which
is what the 256 MB warning counts.

### Sizes

Measured with the defaults (24-texel patches, quality 85):

| Reconstruction | Points | Images | `scene.json` | Patch pages | Thumbnail pages | Total | Decoded atlases |
|----------------|--------|--------|--------------|-------------|-----------------|-------|-----------------|
| Seoul bull ground truth (bitmaps rendered from photographs) | 280, 266 patches | 17 | 44 KB | 57 KB | 114 KB | 239 KB | 2.0 MB |
| Kerry Park tk107 (fisheye) | 384, 377 patches | 48 | 96 KB | 73 KB | 289 KB | 483 KB | 4.1 MB |
| Dino dog toy, 85 images, as embedded patches (bitmaps rendered) | 18,443, 18,426 patches | 85 | 1.2 MB | 3.1 MB (one 3536² page) | 409 KB | 4.7 MB | 53.5 MB |
| The same with `--patch-size 12` | | | 1.2 MB | 1.3 MB (one 1904² page) | 409 KB | 2.9 MB | 19.6 MB |
| OmniHilltop, a 12-camera rig | 46,242, 46,238 patches | 156 | 2.9 MB | 6.1 MB (two pages) | 744 KB | 9.7 MB | 129.5 MB |
| Dino Ledge with `--max-points 150000 --patch-size 12` | 150,000 of 530,674 | 500 | 10.3 MB | 12.5 MB (two pages) | 4.7 MB | 27.5 MB | 144.8 MB |

The patch mosaic runs at about 2 bits per texel at 24 texels and 3 at 12, so a
full 4096 page of 24-texel patches (157 by 157 cells, 24,649 patches) is about
4 MB. `scene.json` costs about 65 bytes per point with patches and 30 without,
so 100,000 points with 24-texel patches come to about 6.5 MB of `scene.json`
and 17 MB of patch pages over five pages: inside the artifact limits of 16 MB
per text file, 15 MB per binary file and 64 MB per version, but past the 256 MB
decoded warning. `scene.json` reaches the 16 MB text-file limit at about 240,000
points, so Dino Ledge's 530,674 points do not fit whole; `--max-points` keeps
the most-observed points and `--patch-size 12` quarters the patch pages. As
single files, Kerry Park tk107 is 604 KB and the dino 5.9 MB.

**On a phone** (WebGL2, texture limit 16,384, a 120 Hz display), each view
drawn in a claude.ai artifact: Kerry Park tk107 and the dino draw
at the display's 120 frames per second, and Dino Ledge's 150,000 patches at 50
to 60, rising when either the patches or the points are turned off.

## The viewer

[`web-export.js`](../../../src/sfmtool/web_export/web-export.js) exports one
function:

```js
import { mount } from "./web-export.js";
const view = await mount(document.getElementById("scene"), "scene.json", {
  showPatches: true, showPoints: true, showCameras: true, background: "#202020",
});
```

`mount(element, scene, options)` takes the URL of a `scene.json`, or the parsed
scene itself (atlas files then resolve against `options.base`), and draws into
the element. Its options are `showPatches`, `showPoints`, `showCameras`,
`background`, `startImage` (open through this camera), `startActive`, `showStats` (a
frame-rate readout in the corner), `fillsPage` (the view is its whole page, as
in `index.html`; see § "On a phone") and
`onEvent(key, value)` for load and frame-rate reports. It returns `{ scene,
stats, setActive, setFull }`.

[`index.html`](../../../src/sfmtool/web_export/index.html) is a full-window
standalone page that mounts active and reads its options from its query string
(`?patches=0&points=0&cameras=0&start=<image name>&bg=<colour>&stats=1`,
the last showing a frame-rate readout in the view's corner), so an iframe
chooses them with no script in the host page. An artifact either iframes
`scene/index.html` or imports `scene/web-export.js` and mounts into its own
element; the second keeps the view in the page's own layout. A page that
iframes views calls `hostFrames()` from the module once, so their Expand works
(§ "On a phone"). A `--single-file`
page carries the scene in a `<script type="application/json" id="wx-scene">`
element, with the atlas pages as data URIs, and the viewer in a
`<script type="text/plain" id="wx-module">` element that `index.html` imports
through a blob URL.

What it draws, each matching SfM Explorer's rule:

- **Patches**: one instanced mesh per atlas page, corners `centre ± u ± v`,
  textured from the atlas, a patch seen from behind not drawn.
- **Points without a patch, and every point when patches are off**: round
  camera-facing splats `2 × point_size` across.
- **Points at infinity**: 4-pixel splats projected with `w = 0`, so they keep
  their direction as the camera moves, drawn behind everything finite.
- **Cameras**: frustum wireframes from the grid, and each image's thumbnail on
  the grid surface. The `start_image` camera's own frustum is hidden while the
  eye is within two frustum lengths of it, since the view opens inside it.

What it does: orbit, pan and zoom about a target with Z up, by mouse and by
touch; a double-click or double-tap on a point or camera re-centres the orbit
there; hovering with a mouse, or tapping, shows a point's index and track
length, or a camera's image name; toggles for patches, points and cameras; a
button that returns to the initial view. It has none of SfM Explorer's
eye-dome lighting, selection or editing.

### On a phone

**Gestures.** One finger drags to orbit, two fingers pinch to zoom and drag
together to pan, and a double-tap re-centres the orbit. These are three.js's
OrbitControls with `ONE: ROTATE, TWO: DOLLY_PAN`. A tap does what hovering does
with a mouse: the label shows until the next tap.

**Scrolling the page around the view.** An inline view in a longer page starts
inactive: its canvas takes `touch-action: pan-y pinch-zoom`, so a one-finger
drag scrolls the page past it, and a "Tap to explore" label covers it. A tap
activates it, the canvas takes `touch-action: none` and every gesture goes to
the view, until the reader taps outside it. Expand fills the window and activates
the view; Close, or Escape, restores it. A view mounted into an element of a page
does this with CSS fixed positioning, not the Fullscreen API, which iPhones
offer only for video, and does so whether or not that page is itself in a frame:
a claude.ai artifact page always is. A view that is the whole page of an iframe
(`index.html`, which mounts with `fillsPage`) cannot grow past the iframe's box,
so it asks the page that holds it: it posts a greeting to its parent, a parent running `hostFrames()`
answers, and from then on Expand and Close post messages that pin the iframe to
the window and release it. A view whose parent does not answer uses the
Fullscreen API when the iframe allows it (`allow="fullscreen"`), and otherwise
shows no Expand button rather than one that does nothing. `index.html` mounts active, since it has no page to scroll.

**Controls and limits.** Every button is at least 44 CSS pixels square, in one
wrapping row at the top right. The canvas renders at a device pixel ratio of at
most 2. The viewer reads `MAX_TEXTURE_SIZE` and, when a patch page is larger,
draws splats with a notice; a thumbnail page that is too large is left out.
Without WebGL2 the element shows a message and the first camera's thumbnail,
cut out of the atlas with CSS.

This behaviour has been checked in a claude.ai artifact on desktop Firefox and
on a phone, where Kerry Park tk107 drew at the display's own 120 frames per
second and the phone reported a maximum texture size of 16,384.

### three.js, imported by full URL

The viewer draws with three.js 0.180.0, pinned in `web-export.js` so an old
export keeps working when the library moves on. It imports the library by full
URL from jsDelivr's `+esm` build,
`https://cdn.jsdelivr.net/npm/three@0.180.0/+esm`, and OrbitControls from
`https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/controls/OrbitControls.js/+esm`,
which jsDelivr rewrites to import that same URL, so one copy of the library
loads. There is no import map: a claude.ai artifact ignores one, because its
page has already started loading modules by the time the page's own content,
and its map, is read. jsDelivr is also one of the two script hosts an artifact
allows.

three.js rather than Babylon.js because the viewer writes its own shaders for
patches, splats and points at infinity either way, so what the library adds is
camera control, texture loading and draw calls, and three.js does that in about
a tenth of the download (0.2 MB compressed against 1.6 MB).

## Code

- **Core**: [`sfmtool_core::web_export`](../../../crates/sfmtool-core/src/web_export.rs)
  builds everything scene-specific. `build_web_export(recon, &options, progress)`
  returns the files in memory with a `WebExportReport`;
  `write_web_export(recon, out_dir, &options, progress)` writes them and returns
  the report. `WebExportOptions` carries the options above, plus `source_name`,
  `generator` and `max_page_size` (4096 unless a caller packs smaller pages).
  The report lists each file with its size, the point, patch and camera counts,
  where the thumbnail rows came from, whether the bitmaps were rendered, the
  decoded atlas bytes and the warnings. Atlas packing is
  [`web_export::atlas`](../../../crates/sfmtool-core/src/web_export/atlas.rs).
  The shared pieces it calls are the display thumbnail row
  (`reconstruction::thumbnail::display_thumbnail_row`), the display patch bitmap
  render (`patch::display_bitmaps::render_display_patch_bitmaps`), the scene
  statistics (`analysis::scene_scale`) and the frustum grid
  (`camera::frustum::compute_distorted_frustum_grid`), each the one SfM Explorer
  uses.

  ```rust
  let options = WebExportOptions { max_points: Some(50_000), ..Default::default() };
  let report = write_web_export(&recon, Path::new("site"), &options, &Progress::none())?;
  ```

- **Binding**: `sfmtool._sfmtool.io.write_web_export(recon, out_dir, *, patches,
  thumbnails, patch_size, jpeg_quality, max_points, start_image, source_name,
  generator)`, returning the report as a dict.
- **Python**: the command in
  [`_commands/web_export.py`](../../../src/sfmtool/_commands/web_export.py), and
  in [`sfmtool.web_export`](../../../src/sfmtool/web_export/__init__.py) the
  directory checks, the copy of the two static files and the single-file page.

## Not provided

- The page shows no covisibility or track lines, and no patch confidence.
- SfM Explorer has no export of its own; it would need only the core module.
- No glTF is written. glTF points have no size, no notion of a point at
  infinity and nowhere to put per-image names for hover.
- There is no JavaScript test harness in the repository; the viewer is checked
  by hand in a browser.

## Tests

- Rust, in `web_export/tests.rs`: tile placement and edge-copied borders, page
  overflow onto several pages, JPEG page size, a pinhole grid on the flat far
  plane, a fisheye grid on the frustum sphere, byte-block offsets and the
  origin shift back to `f64`, `--max-points` selection, patch resampling, the
  start view and an unknown start image.
- Python, in [`tests/test_web_export.py`](../../../tests/test_web_export.py),
  on the Seoul bull ground truth into a temporary directory: `scene.json`
  against the files written and the `.sfmr`, page sizes, the non-empty
  directory refusal and `--overwrite`, leaving out atlases, the start image,
  the single-file page and its 16 MB refusal.
