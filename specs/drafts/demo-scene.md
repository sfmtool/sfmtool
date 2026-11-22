# Demo Scene

**Status:** Draft. Decided by the maintainer (2026-11-21): the reconstruction
is an `embedded_patches` one, with keypoints inline, patch bitmaps rendered
from the scene's photographs and each image's hash taken over its rendered
file, and no `.sift` file or invented descriptor; *Load Demo Data* writes a
small workspace to a temporary directory and opens it from there, and the value
without files stays available to tests; the generator sits beside the sphere
demo, `SfmrReconstruction::demo`, and does not replace it. Not decided: the
questions in [Open questions](#open-questions).

## Purpose

SfM Explorer and the tests need a reconstruction whose every number is known
to be right. A reconstruction solved from real photographs is never known to be
right: its poses, points and normals are estimates, and a test built on one can
only say that a result did not change. This draft proposes a demo
reconstruction built in the opposite order. A small 3D scene is defined first, a
textured ground plane with a cube standing on it; photographs of that scene are
rendered from five pinhole cameras; and the reconstruction is written down from
the scene: the cameras' poses, 3D points sampled on the surfaces, the pixel
where each camera sees each point, which cameras do not see it because the cube
is in the way, each point's surface normal and patch, and the patch bitmaps
rendered from the photographs. Every one of those is exact by construction, and
the photographs agree with them, so the result is a whole reconstruction with
ground truth.

The scene is chosen to hold the cases where the toolkit's estimators fail:
creases, where a patch straddles two faces of the cube or a face and the
ground; occlusion edges, where a ground point is seen by some cameras and
hidden behind the cube from others; and grazing views of the ground. A normal
estimator, a track check or a bitmap renderer can be run on it and its answer
compared against the stored truth.

It has two consumers. SfM Explorer's *File > Load Demo Data* writes the scene
as a workspace and opens it, so every viewer feature that reads photographs
(Image Detail, the bench, display patch bitmaps, *Build SIFT Files*) works on
the demo. Tests take the reconstruction value and the rendered photographs in
memory, with no files.

The existing demo, `SfmrReconstruction::demo` in
[demo.rs](../../crates/sfmtool-core/src/reconstruction/data/demo.rs), stays: a
relaxed sphere of points seen by a ring of eight cameras, with no photographs,
invented readings and `sift_files` observations. It is cheap and has no pixels,
which is what most of its roughly 300 call sites, nearly all of them tests,
want. The demo scene is the second generator, for the tests and the viewer
that need pixels.

## Rust API

The generator would live in
`crates/sfmtool-core/src/reconstruction/data/demo_scene.rs`, beside
[demo.rs](../../crates/sfmtool-core/src/reconstruction/data/demo.rs), with
submodules for the renderer and the texture in `demo_scene/`.
`reconstruction::data` is private to the crate, so the types are re-exported
from `sfmtool_core::reconstruction`, as `SfmrReconstruction` is. They are not bound
to Python in this proposal (see [Open questions](#open-questions)).

```rust
/// How the demo scene is rendered and sampled. `Default` is the scene this
/// spec describes.
#[derive(Debug, Clone, PartialEq)]
pub struct DemoSceneOptions {
    /// The photographs' size in pixels: 640 x 480.
    pub image_width: u32,
    pub image_height: u32,
    /// The pinhole focal length in pixels, the same on both axes: 560.
    pub focal_length_px: f64,
    /// Samples per pixel along each axis; the renderer takes the square of
    /// this many rays per pixel: 2 (4 rays).
    pub supersampling: u32,
    /// The spacing of the point grid on the ground, in metres: 0.25.
    pub ground_point_spacing: f64,
    /// The spacing of the point grid on each cube face, in metres: 0.125.
    pub face_point_spacing: f64,
    /// Each point's patch half-extent, in metres: 0.09.
    pub patch_half_extent: f64,
    /// The patch bitmaps' side `R`: 24.
    pub patch_bitmap_resolution: u32,
    /// A camera that views a point's patch more than this many degrees off
    /// its normal does not observe it: 80.
    pub max_obliquity_deg: f64,
    /// Picks the surfaces' textures and the points' offsets from the grid.
    pub seed: u64,
}

/// The rendered scene: its cameras, poses and photographs, in memory.
pub struct DemoScene { /* … */ }

impl DemoScene {
    /// Render the scene's photographs, one per camera, in a phase `render`
    /// that counts rows.
    pub fn render(
        options: &DemoSceneOptions,
        progress: &Progress<'_>,
    ) -> Result<DemoScene, DemoSceneError>;

    /// The options the scene was rendered with.
    pub fn options(&self) -> &DemoSceneOptions;
    /// Each photograph's image name, `images/demo_0.png` to `images/demo_4.png`.
    pub fn image_names(&self) -> &[String];
    /// The rendered photographs, RGB, in image order.
    pub fn photographs(&self) -> &[ImageU8];
    /// The photographs as the tiled sources the patch kernels read.
    pub fn tiled_photographs(&self) -> Vec<TiledPhotograph>;
    /// Each photograph's encoded file, the bytes `write_workspace` writes and
    /// `image_file_hashes` hashes.
    pub fn encoded_photographs(&self) -> &[Vec<u8>];

    /// The reconstruction of the scene, with no files behind it
    /// (`workspace_dir` empty): sample the points, find who sees each one,
    /// and render the patch bitmaps from the photographs, in phases `points`
    /// and `patch bitmaps`.
    pub fn reconstruction(
        &self,
        progress: &Progress<'_>,
    ) -> Result<SfmrReconstruction, DemoSceneError>;

    /// Write the scene as a workspace in `dir`, which must be empty or not
    /// exist: the photographs under `images/`, a `.sfm-workspace.json`, and
    /// `demo.sfmr` at the workspace root. Returns the `.sfmr` path.
    pub fn write_workspace(
        &self,
        dir: &Path,
        progress: &Progress<'_>,
    ) -> Result<PathBuf, DemoSceneError>;
}

#[derive(Debug)]
pub enum DemoSceneError {
    /// An option out of its range, naming it and the range.
    Options(String),
    /// `progress` was cancelled.
    Cancelled,
    /// `dir` holds files, or lies inside another workspace.
    Directory(String),
    /// A photograph could not be encoded or written.
    Photograph(image::ImageError),
    /// The `.sfmr` could not be written.
    Save(SfmrError),
    Io(std::io::Error),
}
```

**Why render, then build, then write.** The three steps have different
callers. A kernel test wants the photographs in memory and the value, and no
disk: it calls `render` and `reconstruction`. *Load Demo Data* wants all three.
A test of the renderer alone, or a bench test that builds its own points on the
scene's photographs, wants only `render`. The value cannot be built without the
photographs, because the patch bitmaps, the thumbnails, the image readings and
the image file hashes all come from them, so the rendered `DemoScene` is what
`reconstruction` and `write_workspace` are methods of, and the photographs are
rendered once whichever of the two a caller needs.

**Why one options struct.** The scene's geometry (the plane, the cube, the
camera positions) is fixed, and is what the tests reason about; the options are
the parts a caller has a reason to change: a smaller image or fewer points for
a cheaper test, a larger bitmap side, no supersampling to test aliasing. It is
named `*Options` by [GLOSSARY.md](../GLOSSARY.md) § "Rust type names": the
caller fills it with choices and it has a `Default`.

**Why errors, not panics.** Options come from callers, and
[STANDARDS.md](../STANDARDS.md) § "Caller input never panics" applies: a zero
image size, a zero supersampling, a non-finite or non-positive spacing,
half-extent or focal length, a bitmap side of zero, or an obliquity outside
`0..=90` is `DemoSceneError::Options`. An option set that leaves some point
with fewer than two observations is not an error; that point is left out.

**Why `&Progress`.** Each of the three walks every image, row or point, which
[STANDARDS.md](../STANDARDS.md) § "Long-running functions take a Progress"
counts as long-running; `write_workspace` also writes a whole `.sfmr`. The
viewer runs them as a background task it can cancel.

Example, a test that needs the scene's value and photographs:

```rust
use sfmtool_core::progress::Progress;
use sfmtool_core::reconstruction::{DemoScene, DemoSceneOptions};

let scene = DemoScene::render(&DemoSceneOptions::default(), &Progress::none())?;
let recon = scene.reconstruction(&Progress::none())?;
let photographs = scene.tiled_photographs();
// recon.point_set.points[i].normal is the surface's normal exactly, and
// every keypoint is its point's projection rounded to f32.
```

And *Load Demo Data*, on a background thread:

```rust
let scene = DemoScene::render(&DemoSceneOptions::default(), &progress)?;
let sfmr_path = scene.write_workspace(&temp_dir, &progress)?;
// then the same open as File > Open, of sfmr_path
```

## The scene

The world is the `.sfmr` convention
([sfmr-file-format.md § Coordinate System Conventions](../formats/sfmr-file-format.md#coordinate-system-conventions)):
right-handed, +Z up, the X-Y plane the ground, in metres, and the file records
`world_space_unit: "m"`.

- **Ground.** The plane `z = 0` over the square `[-5, 5] × [-5, 5]` m, seen
  from above only.
- **Cube.** Side 1 m, standing on the ground with its base centred on the
  origin, so it occupies `z ∈ [0, 1]`. It is turned about +Z so that its faces
  are not square to the camera arc: its horizontal axes are `a = (0.8, 0.6, 0)`
  and `b = (-0.6, 0.8, 0)`, a turn of about 36.9° written with no
  trigonometric function. Its six faces have outward normals `±a`, `±b`, `+z`
  (top) and `-z` (the base, which rests on the ground and is not seen).
- **Background.** A ray that meets neither surface sees one flat colour. No
  point lies on it.
- **No lighting.** A surface's colour is its texture alone, the same from
  every direction, so a surface point has the same colour in every photograph
  that sees it, up to the renderer's sampling and the rounding to 8 bits. This
  is what makes the patch bitmaps of one point agree between views, and it is
  the assumption every photometric score in the toolkit makes.

### Texture

Each surface carries value noise: pseudo-random levels on a square lattice,
blended between lattice nodes with the smoothstep `3t² − 2t³`, so that no shift
of the texture resembles another and a patch can be located by its content.
The ground and each cube face use a different seed, so the cube does not blend
into the ground, and two faces meeting at an edge do not continue each other's
pattern. The noise is evaluated in each surface's own 2D coordinates: `(x, y)`
on the ground, and on a cube face the coordinates along its two in-face axes
from one corner.

The lattice spacing is 0.02 m. A surface about 4 m from a camera with a focal
length of 560 px is about 7 mm per pixel, so a lattice cell spans about 2.8 px
in a view that faces it: detail at the scale of a patch, as the bench tests
choose it (their plane texture is about 2.5 px a cell in a facing view,
[grown_plane_normal.rs](../../crates/sfmtool-core/src/bench/tests/grown_plane_normal.rs)).
The far ground and the ground at a grazing angle have cells under 2 px across,
which the supersampling averages rather than aliases.

The noise function is the one the bench tests already use, moved into the
library and shared rather than copied a third time. There are two copies today,
`noise` in
[grown_plane_normal.rs](../../crates/sfmtool-core/src/bench/tests/grown_plane_normal.rs)
(seeded, in `128 ± amplitude`) and `other_texture` in
[grid_plane_normal.rs](../../crates/sfmtool-core/src/bench/tests/grid_plane_normal.rs)
(the same hash with no seed term, in `30 + 195 · level`). They are one function
of `(x, y, spacing, seed)` returning a value in `[-1, 1]`: the unseeded hash is
the seeded one at seed 0, since the seed enters by an XOR with
`seed · constant`, and `30 + 195 · l` is `127.5 + 97.5 · (2l − 1)`. Each caller
maps the shared value to its own range. The two forms are equal in exact
arithmetic and can differ in the last bit in `f64`, which the bench tests'
assertions, all tolerances, do not see.

**Colour.** Each surface has its own tint, and its colour is the tint scaled by
`0.5 + 0.4 · noise`, with one noise channel per surface. A grey scene would
serve every kernel the same, since they read grey; the tint is for the person
looking at the viewer, who should see from its colour which face a patch is on.
Whether to tint is an open question.

## The cameras

One shared pinhole camera, 640 × 480 px, focal length 560 px on both axes and
the principal point at the centre `(320, 240)`: a horizontal field of view of
about 59°. No distortion and no per-image adjustment.

Five cameras on an arc 4 m from the Z axis, on the side of the cube toward
−Y, at azimuths −55°, −27.5°, 0°, 27.5° and 55° from −Y, so the arc spans
110°. They alternate between two heights, 1.2 m (the first, third and fifth)
and 3.0 m (the second and fourth). Each looks at the cube's centre
`(0, 0, 0.5)` with its +Y axis in the vertical plane through its line of sight,
upward, as `demo.rs` builds its poses: the rows of the world-to-camera rotation
are the camera's right, up and back axes in world coordinates, and the stored
pose is the canonical world-to-camera one, `p_cam = R · p_world + t`, the
camera looking down −Z with +Y up. The camera centres are written as decimal
literals (for instance `(-3.277, -2.294, 1.2)`, `(0, -4, 1.2)`), not computed
with `sin` and `cos`, so the poses are built from additions, multiplications,
divisions and square roots alone, which IEEE 754 rounds the same on every
platform.

What each camera sees, with the cube's side faces named by their outward
normals:

| Camera azimuth | Height | Face `-b = (0.6, -0.8, 0)` | Face `-a = (-0.8, -0.6, 0)` | Top |
|---|---|---|---|---|
| −55° | 1.2 m | about 99° off its normal: not seen | about 12°: facing | about 87°: past the 80° cut |
| −27.5° | 3.0 m | about 74° | about 44° | about 63° |
| 0° | 1.2 m | about 43° | about 60° | about 87°: past the cut |
| 27.5° | 3.0 m | about 37° | about 88°: past the cut | about 63° |
| 55° | 1.2 m | about 23° | not seen | about 87°: past the cut |

(The angle between the face's outward normal and the direction from the
face's centre to the camera; points away from the centre differ by a few
degrees.) The face `-b` is observed by four cameras, from 23° to 74°; the face
`-a` by three, and seen by a fourth at 88°, past the cut. The top is observed only by
the two high cameras, which is the fewest a point can have, and is seen almost
edge-on by the low ones. The ground is seen at a grazing angle by the three low
cameras, about 12° to 19° above the ground around the cube. The face `+a` is
seen only by the camera at 55°, about 79° off its normal and so just inside the
cut, and `+b` by no camera, so no point on either has two observations and none
is kept.

## The renderer

Every pixel is the mean of `supersampling²` rays through a regular grid of
sample positions inside it, at offsets `(i + 0.5) / s` for `i` in `0..s` along
each axis. Each ray starts at the camera centre with the direction
`R_iᵀ · pixel_to_ray(u, v)` (`CameraIntrinsics::pixel_to_ray`, which for a
pinhole is `((u − cx) / f, −(v − cy) / f, −1)` normalised), and is intersected
with:

- the cube, by the slab test in the cube's own axes `a`, `b`, `z`: the ray's
  origin and direction are expressed in those axes by dot products, and the
  entry distance is the largest of the three slabs' near distances when it is
  below the smallest far distance;
- the ground, at `λ = −C_z / d_z` where `d_z < 0`, when the hit lies inside
  the square.

The nearest hit with a positive distance gives the surface and the point, and
the surface's texture gives the colour; no hit gives the background. The
samples are averaged in `f32` and rounded to 8 bits. The geometry is `f64`: the
scene's coordinates reach 10 m and a texture coordinate has to resolve a
fraction of a 2 cm cell, about `10⁻⁴` of the coordinate's size, which `f32`'s
`6 · 10⁻⁸` holds but leaves no margin through the slab test's divisions; the
averaging of at most a few dozen 8-bit samples needs `f32` at most
([STANDARDS.md](../STANDARDS.md) § "Numeric precision is the smallest the use
justifies").

**Determinism.** The output depends only on the options. The texture's levels
come from the hash of the lattice node and the seed, with no random number
generator, and the renderer uses `+`, `−`, `×`, `÷`, `sqrt` and `floor`, so a
photograph is the same bytes on every platform. Rows are rendered in parallel
with rayon, each row written by one thread and summed in a fixed order, so the
thread count does not change a pixel.

**Cost.** Five 640 × 480 photographs at four rays a pixel are 6.1 million rays.
A ray is one slab test, one plane test and one evaluation of the noise (four
hashes and a bilinear blend), on the order of 50 to 100 ns, so 0.3 to 0.6 s on
one thread and well under 0.1 s across 16 threads. The bitmaps are cheaper
still: a whole-cloud render of about 1,000 points takes 0.1 to 0.2 s
on 32 threads ([reference-view.md § Cost of the stored bitmap](../core/patch/reference-view.md#cost-of-the-stored-bitmap)).
The step plan measures both.

## The points and who sees them

**Sampling.** Points lie on a grid on each surface, each moved from its grid
position by a seeded offset of up to a quarter of the spacing along each axis,
so the points do not line up into rows that a regular pattern in the
photographs would echo:

- On the ground, a grid at `ground_point_spacing` (0.25 m) over
  `[-2.5, 2.5] × [-2.5, 2.5]`, leaving out the points inside the cube's
  footprint: about 420 grid points, of which about 350 are observed by two
  cameras or more. The rest lie near the cameras, below the views' lower
  edges, or at the grid's corners outside the outer views.
- On each cube face but the base, a grid at `face_point_spacing` (0.125 m)
  whose first row is half a spacing from the edge: 64 points a face. Only the
  top and the faces `-a` and `-b` keep points, about 190.

About 540 points at the defaults. With a patch half-extent of 0.09 m, the
patches of the outer row on each face cross the cube's edge, and those of the
ground points next to the cube run into its side: those are the crease cases.
The ground points behind the cube, seen past it by the outer cameras and hidden
from the middle ones, are the occlusion cases.

**Visibility.** Camera `i` observes point `X` on a surface with normal `n` when
all of these hold:

1. `X` is in front of the camera (camera-space `z < 0`) and its projection lies
   inside the image, `[0, 640) × [0, 480)`;
2. the camera is on the side `n` faces, and views the patch no more than
   `max_obliquity_deg` (80°) off `n`: `n · (C_i − X) / |C_i − X| ≥ cos 80°`, the
   same cut `sfm embed-patches` applies by default
   ([embed-patches-command.md](../cli/reconstruction/embed-patches-command.md));
3. the segment from the camera centre `C_i` to `X` does not meet the cube
   before it reaches `X`: the slab test of the renderer, with the entry
   distance compared against the distance to `X` less `10⁻⁹` of it, so that a
   point on the face the ray enters is not hidden by its own face.

The ground cannot hide a cube point, since every camera is above it, and a
cube face cannot hide a point on a face that looks toward the camera, since the
cube is convex; the third test is what decides which ground points the cube
hides. A point that fewer than two cameras observe is left out.

## The reconstruction's columns

The value is an `embedded_patches` reconstruction
([sfmr-file-format.md § Observation source](../formats/sfmr-file-format.md#observation-source)),
the observation mode the maintainer chose: it needs no `.sift` file, and a
`.sift`'s descriptors could only be invented, since no detector found these
points. Each column, and why it is exact:

| Column | Value | Exact because |
|---|---|---|
| `cameras` | the one pinhole above | stated |
| `images/camera_indexes` | `0` for every image | |
| `images/quaternions_wxyz`, `translations_xyz` | each camera's world-to-camera pose | built from the stated centres and target; `f64` |
| `images/names` | `images/demo_0.png` … `images/demo_4.png` | |
| `images/image_file_hashes` | XXH128 of each encoded photograph | the bytes `write_workspace` writes |
| `images/thumbnails_y_x_rgb` | each photograph through `thumbnail_from_rgb` | the format's area resize of the actual photograph |
| `images/sharpness`, `gradient_structure_tensors_xx_yy_xy`, `image_to_gray` | `image_readings_from_rgb` of each photograph under `ImageToGray::default()` | computed from the photograph, as a writer that gathers them from the image file would ([sfmr-file-format.md § Image readings](../formats/sfmr-file-format.md#image-readings), rule 3) |
| `points3d/positions_xyzw` | the sampled points, `w = 1` | sampled on the surface; `f64` |
| `points3d/colors_rgb` | the texture's colour at the point | evaluated, not estimated |
| `points3d/reprojection_errors` | each point's RMS reprojection error, scored by `recompute_point_errors` | what the stored keypoints give: about `10⁻⁵` px, the rounding of the keypoints to `f32` |
| normals (`has_normals`) | `+z` on the ground, the face's outward normal on the cube | derived from the patch frame, `normalize(u × v)` in `f32`, as the format defines it ([sfmr-file-format.md § Derived normals](../formats/sfmr-file-format.md#derived-normals)) |
| `points3d/patch_u_halfvec_xyz`, `patch_v_halfvec_xyz` | a square patch frame of half-extent 0.09 m in the surface: on the ground `u = h·x̂`, `v = h·ŷ`; on a side face with normal `n`, `v = h·ẑ` and `u = h·(ẑ × n)`; on the top `u = h·a`, `v = h·b` | each makes `u × v` point along the surface's outward normal, toward the cameras that see it |
| `tracks/image_indexes`, `point_indexes`, `observation_counts` | the observations visibility keeps, sorted by `(point, image)` | the ray–box test is exact |
| `tracks/keypoints_xy` | each point's projection into the image, `f32` | projected through the camera; the keypoint is where the photograph sees the patch's centre, since the photograph was rendered from the same pose |
| `tracks/reference_observations` | the reference-view rule's pick for each point | the same render `--add-patch-bitmaps` runs |
| `points3d/patch_bitmaps_y_x_rgba` | each point's reference observation's `24 × 24` render | rendered from the photographs by `render_patch_cloud_bitmaps` with `UnreferencedPoints::Pick` |
| `derived/` depth statistics | `recompute_depth_statistics` | computed from the poses and points |
| `metadata` | `operation: "demo"`, `world_space_unit: "m"`, `feature_source: "embedded_patches"` | |

Absent: rigs and frames, per-image adjustments, normal confidence, point
constraints, observation confidence, observation readings and measurements.

**The bitmaps come from the photographs, not from the texture.** The format
says a point's bitmap is the render of its reference observation from that
photograph, so that dropping the bitmaps and adding them again gives the same
bitmaps ([sfmr-file-format.md § `tracks/reference_observations`](../formats/sfmr-file-format.md#tracksreference_observationsnint32zst-with-the-patch-frame)).
Sampling the texture over the patch instead would give a sharper bitmap that no
later render reproduces. The generator renders them through the same function
the viewer's display render and `--add-patch-bitmaps` reach
([stored_bitmap.rs](../../crates/sfmtool-core/src/patch/stored_bitmap.rs)),
with the photographs in memory as `TiledPhotograph`s. The texture is then a
check on the renderer: a test can compare a facing point's bitmap with the
texture sampled over its patch.

**The in-memory value and the written file are the same reconstruction.** The
photographs are written as PNG, which is lossless, so a photograph decoded from
the workspace is the pixels the bitmaps, thumbnails and readings were computed
from, and the image file hashes are of the bytes on disk. A save of the value
therefore needs nothing from disk: its readings are its own rows (rule 1 of
"Where a writer gets the values"), and a load of the written `.sfmr` gives the
value back with the workspace directory filled in. A JPEG would break this: its
decoded pixels differ from the rendered ones, so the in-memory bitmaps would
not be what a render from the workspace gives.

## The workspace

`write_workspace(dir)` writes, into an empty or missing `dir`:

```
dir/
├── .sfm-workspace.json
├── demo.sfmr
└── images/
    ├── demo_0.png
    ├── …
    └── demo_4.png
```

The marker is a version 2 workspace file for the `sfmtool` feature tool with
its default options, its `image_to_gray` the reconstruction's (BT.709 luma,
unrounded), and its `feature_prefix_dir` named from them as `sfm ws init` names
it ([workspace.md § The `.sfm-workspace.json` File](../workspace/workspace.md#the-sfm-workspacejson-file)).
An `embedded_patches` file needs no `feature_prefix_dir`, but recording the
`sfmtool` tool lets *Build SIFT Files*, the SIFT index and every command that
extracts features run on the demo workspace as on any other. The `.sfmr` sits
at the workspace root and records `workspace.relative_path` as `.`, so the
workspace moves with it. `dir` inside another workspace is refused, as `sfm ws
init` refuses to nest one
([workspace.md § Protection Against Nesting](../workspace/workspace.md#protection-against-nesting)).

**What the code does not do yet.** No Rust code writes a `.sfm-workspace.json`:
`init_workspace` is Python
([_workspace.py](../../src/sfmtool/_workspace.py)), and the `sfmtool` tool's
default `feature_options` are built in Python
(`get_default_sfmtool_feature_options`). Rust can name the folder
(`feature_prefix_dir` in
[workspace/feature_options.rs](../../crates/sfmtool-core/src/workspace/feature_options.rs))
and read the options into `SiftParams` (`SiftParams::from_feature_options`),
but cannot produce the default options. Python's defaults are not `SiftParams::default()`: they set
`double_image_below_long_side` to 4096, where the Rust default has no limit.
The first step of the plan adds a Rust function that writes a version 2 marker
for the `sfmtool` tool with the options Python writes today, which Python's
`init_workspace` then calls, so the two cannot drift. `ImageU8::write` writes to a path only; computing the hash of a
PNG without writing it needs an encoder to bytes beside it
([reading-and-writing-images.md](../core/camera/reading-and-writing-images.md)).
The workspace marker's file name is defined three times in Rust
(`WORKSPACE_MARKER` in `sfmtool-archive-io`, `WORKSPACE_CONFIG_FILE` in
`features::sift::files` and `WORKSPACE_FILE_NAME` in `workspace`); the writer
uses the first.

## Load Demo Data

Today *File > Load Demo Data...* opens a dialog asking for a number of points
and appends a node holding `SfmrReconstruction::demo(n)` with no path
(`AppState::load_demo` in [ops.rs](../../crates/sfm-explorer/src/state/ops.rs),
the dialog in [modals.rs](../../crates/sfm-explorer/src/app/modals.rs)). With
this proposal it:

1. starts a background task, as *File > Open* does, that renders the scene,
   creates a new directory under the system's temporary directory (for
   instance `sfmtool-demo-<process id>-<n>`) and calls `write_workspace` into
   it, reporting through the task's `Progress`;
2. opens the written `demo.sfmr` by the same open as *File > Open*
   (`AppState::start_open` in
   [open.rs](../../crates/sfm-explorer/src/state/open.rs)), so the node is
   labelled `demo` from the file's stem and its workspace, photographs and
   photograph cache are those of any opened file;
3. deletes the directory when the viewer exits.

The node now has a path, so several things the viewer and its specs say of
demo data change: `path` is the temporary file rather than `null` on the wire
([mcp-server.md](../gui/mcp-server.md)); *Save* on it would write into a
directory that is deleted at exit ([saving.md](../gui/saving.md)); `--demo`,
which `ui_basic` uses to set a viewer up
([architecture.md § Testing](../gui/architecture.md#testing)), becomes a
background open that a test waits on, as it waits on a file named on the
command line; and the dialog's *Number of points* box, which the `type_text`
test types into, has no number to set unless the dialog keeps one. Each is an
[open question](#open-questions) or a step of the plan.

## Tests that could move to it

The tests that build pixels or patch geometry by hand on top of the sphere
demo are the candidates, because the scene gives them real photographs that
agree with the value:

- SfM Explorer's `embedded_demo` and `projected_embedded_demo`
  ([state/edits/tests.rs](../../crates/sfm-explorer/src/state/edits/tests.rs)),
  which convert the sphere demo to `embedded_patches` with invented keypoints
  or patch frames turned toward the cameras and no photographs, used by the bench,
  Track View, Image Detail and MCP bench tests. On the scene, a bench step such
  as Grown Plane Normal or track-at-pixel runs on photographs and its answer
  can be checked against the stored normal.
- The display thumbnail, Image Detail and *Build SIFT Files* tests, which today
  have no photograph to read.
- `web_export`'s `far_demo`, which needs patch frames and bitmaps.
- The bench kernel tests' `fixture_of`
  ([bench/tests/scene.rs](../../crates/sfmtool-core/src/bench/tests/scene.rs)),
  which starts from `SfmrReconstruction::demo(1)` and replaces every column. Its
  scenes (a plane facing the cameras at 128 × 128 px, steps between depths)
  are chosen per test and stay; what it can share is the value noise and the
  construction of an `embedded_patches` value from a scene.

The tests that only want a cheap value with no pixels keep the sphere demo: the
3D view's upload tests, the reconstruction data and per-image adjustment tests,
the infinity conversion, alignment and reprojection-noise tests, and most of
the viewer's scene graph and save tests. Each migration is decided test by
test, by whether the test's assertions gain from real pixels; a test that gains
nothing from them stays on the sphere demo.

## Open questions

1. **Image file format.** PNG keeps the in-memory value and the written file
   identical (see above), at about 0.6 to 0.9 MB a photograph, since noise
   compresses poorly: under 5 MB for the workspace. JPEG would be a fifth of
   that and would make the in-memory value differ from a load of the file. The
   draft proposes PNG.
2. **Point count and spacing.** About 540 points at the defaults. More points
   make a denser cloud to look at and cost little; fewer keep the tests fast.
   Should the viewer keep a *Number of points* setting, which would set the
   spacings, or load the defaults with no dialog?
3. **Grey or tinted texture.** The kernels read grey; a tint per surface helps a
   person see which face a patch is on.
4. **Lens distortion, a second camera, a rig.** The scene has one undistorted
   pinhole. A fisheye or a radial B-spline camera, a second camera model, or a
   two-camera rig would exercise more of the toolkit; each is an option to add
   later without changing the default scene.
5. **Where the temporary workspace lives and when it is deleted.** The system
   temporary directory, deleted at exit, is proposed. A viewer that crashes
   leaves its directory behind: should the next start remove `sfmtool-demo-*`
   directories whose process has ended? Two loads in one session write two
   directories, or the second reuses the first?
6. **Saving the demo node.** *Save* writes into the temporary workspace, which
   exit deletes. Should *Save* on a node whose file is in a demo workspace act
   as *Save As*, as it does today for a node with no file? A *Save As* to
   another directory records the temporary workspace's absolute path, whose
   photographs are gone after exit; the file still opens, since an
   `embedded_patches` file with thumbnails and bitmaps needs no photograph, but
   without the photographs. Should Load Demo Data instead offer to write the
   workspace somewhere the user chooses?
7. **A measurement.** The cube's edge is exactly 1 m. A distance measurement
   between two of its corners, seen in two images, would give the
   Measurements panel a known answer. In or out of the first version?
8. **A Python binding.** The Python tests have no demo today. A binding of
   `DemoScene` would give them a workspace with photographs to run CLI commands
   on, at the cost of rendering in each test session.

## Step plan

Each step is one change on its own branch, with its tests.

1. **Renderer and scene.** The shared value noise moved into the library, with
   the two bench tests calling it; the Rust writer of a version 2 `sfmtool`
   workspace marker, which Python's `init_workspace` calls; an `ImageU8` PNG
   encoder to bytes; `DemoSceneOptions` and `DemoScene::render`. Tests: the
   photographs are the same bytes on two runs and across thread counts; a
   facing pixel's colour is the texture at the point its centre ray hits, to
   within the supersampling's blur; the cube's silhouette is where the slab
   test puts it. Measure the render's time.
2. **The reconstruction.** Point sampling, visibility, the columns, the
   bitmaps; `DemoScene::reconstruction` and `DemoScene::write_workspace`.
   Tests: every keypoint is its point's projection; no observation's segment
   meets the cube; a ground point behind the cube is hidden from the middle
   camera and seen from an outer one; the derived normals are the surface
   normals; the value saves, verifies and loads back equal, with the workspace
   directory set; the image file hashes match the written files.
3. **Load Demo Data.** The menu entry and `--demo` load the scene from a
   temporary workspace; the directory is deleted at exit; the gui specs
   ([architecture.md](../gui/architecture.md),
   [mcp-server.md](../gui/mcp-server.md), [saving.md](../gui/saving.md),
   [user-experience.md](../gui/user-experience.md)) and the `ui_basic` tests
   follow. This step changes what the viewer shows, so its PR carries
   screenshots.
4. **Test migration.** The tests listed above, one area at a time, each kept
   only where real pixels make an assertion possible or exact.
