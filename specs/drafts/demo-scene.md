# Demo Scene

**Status:** Draft. Decided by the maintainer (2026-11-21): the reconstruction
is an `embedded_patches` one, with keypoints inline, patch bitmaps rendered
from the scene's photographs and each image's hash taken over its written
file, and no `.sift` file or invented descriptor; *Load Demo Data* writes a
small workspace to a temporary directory and opens it from there, and the value
without files stays available to tests; the generator sits beside the sphere
demo, `SfmrReconstruction::demo`, and does not replace it. Decided by the
maintainer (2026-11-22): the photographs are written as JPEG, so that they
differ a little from the exact render as a real photograph does; *Load Demo
Data* keeps a number of points; each surface has its own tint; the scene has
one undistorted pinhole camera; the temporary workspace lives in the system
temporary directory and is deleted at exit; the demo node saves as any other
node does; the first version holds a 1 m distance measurement along a cube
edge; the demo scene is bound to Python; and decoded JPEG pixels may differ
by one level between x86 and arm64, which the tests tolerate. Steps 1, 2 and
3 of the [step plan](#step-plan) are built; step 1 found that the JPEG encoder's
two paths give different bytes, and step 2 that the reference-view rule's pick
of a point's bitmap view can change with a one-level change in the pixels
([The reconstruction's columns](#the-reconstructions-columns)), which the
tests tolerate, as the maintainer decided (2026-11-22). Decided by the
maintainer (2026-11-22): the encoder keeps its AVX2 path, and the tests store the JPEG hashes and the photograph-derived
values once for each encoder path
([Determinism of the JPEG encoder and decoder](#determinism-of-the-jpeg-encoder-and-decoder)).

## Purpose

SfM Explorer and the tests need a reconstruction whose every number is known
to be right. A reconstruction solved from real photographs is never known to be
right: its poses, points and normals are estimates, and a test built on one can
only say that a result did not change. This draft proposes a demo
reconstruction built in the opposite order. A small 3D scene is defined first, a
textured ground plane with a cube standing on it; photographs of that scene are
rendered from five pinhole cameras and written as JPEG; and the reconstruction
is written down from the scene: the cameras' poses, 3D points sampled on the
surfaces, the pixel where each camera sees each point, which cameras do not see
it because the cube is in the way, each point's surface normal and patch, a
distance measurement of a cube edge, and the patch bitmaps rendered from the
photographs. The geometry is exact by construction, and the photographs agree
with it up to the JPEG compression a real photograph also has, so the result is
a whole reconstruction with ground truth.

The scene is chosen to hold the cases where the toolkit's estimators fail:
creases, where a patch straddles two faces of the cube or a face and the
ground; occlusion edges, where a ground point is seen by some cameras and
hidden behind the cube from others; and grazing views of the ground. A normal
estimator, a track check or a bitmap renderer can be run on it and its answer
compared against the stored truth.

It has three consumers. SfM Explorer's *File > Load Demo Data* writes the scene
as a workspace and opens it, so every viewer feature that reads photographs
(Image Detail, the bench, display patch bitmaps, *Build SIFT Files*) works on
the demo. Rust tests take the reconstruction value and the photographs in
memory, with no files. Python tests take the same through a binding, and can
write the workspace to run CLI commands on.

The existing demo, `SfmrReconstruction::demo` in
[demo.rs](../../crates/sfmtool-core/src/reconstruction/data/demo.rs), stays: a
relaxed sphere of points seen by a ring of eight cameras, with no photographs,
invented readings and `sift_files` observations. It is cheap and has no pixels,
which is what most of its roughly 300 call sites, nearly all of them tests,
want. The demo scene is the second generator, for the tests and the viewer
that need pixels.

## Rust API

The generator lives in
`crates/sfmtool-core/src/reconstruction/data/demo_scene.rs`, beside
[demo.rs](../../crates/sfmtool-core/src/reconstruction/data/demo.rs), with
submodules in `demo_scene/`: the texture, the renderer, the points and their
visibility (`points.rs`), and the reconstruction's columns (`value.rs`).
`reconstruction::data` is private to the crate, so the types are re-exported
from `sfmtool_core::reconstruction`, as `SfmrReconstruction` is.

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

impl DemoSceneOptions {
    /// The number of points the default options keep, which a test of the
    /// sampling fixes.
    pub const DEFAULT_POINT_COUNT: u32;

    /// The default options with both point spacings scaled by
    /// `sqrt(DEFAULT_POINT_COUNT / point_count)`, so the scene keeps about
    /// `point_count` points. A `point_count` of 0 is `DemoSceneError::Options`.
    pub fn for_point_count(point_count: u32) -> Result<DemoSceneOptions, DemoSceneError>;
}

/// The rendered scene: its cameras, poses and photographs, in memory.
pub struct DemoScene { /* … */ }

impl DemoScene {
    /// Render the scene's photographs, one per camera, in a phase `render`
    /// that counts rows; then encode each as a JPEG and decode it again, in a
    /// phase `encode` that counts images.
    pub fn render(
        options: &DemoSceneOptions,
        progress: &Progress<'_>,
    ) -> Result<DemoScene, DemoSceneError>;

    /// The options the scene was rendered with.
    pub fn options(&self) -> &DemoSceneOptions;
    /// Each photograph's image name, `images/demo_0.jpg` to `images/demo_4.jpg`.
    pub fn image_names(&self) -> &[String];
    /// The renderer's exact output, RGB, in image order, before JPEG.
    pub fn rendered_photographs(&self) -> &[ImageU8];
    /// The photographs as a reader of the written files decodes them: each
    /// JPEG of `encoded_photographs` decoded by `ImageU8::decode_rgb`.
    pub fn photographs(&self) -> &[ImageU8];
    /// The decoded photographs as the tiled sources the patch kernels read.
    pub fn tiled_photographs(&self) -> Vec<TiledPhotograph>;
    /// Each photograph's JPEG file, the bytes `write_workspace` writes and
    /// `image_file_hashes` hashes.
    pub fn encoded_photographs(&self) -> &[Vec<u8>];

    /// The reconstruction of the scene, with no files behind it
    /// (`workspace_dir` empty): sample the points, find who sees each one,
    /// place the measurement, and render the patch bitmaps from the decoded
    /// photographs, in phases `points` and `patch bitmaps`.
    pub fn reconstruction(
        &self,
        progress: &Progress<'_>,
    ) -> Result<SfmrReconstruction, DemoSceneError>;

    /// Write the scene as a workspace in `dir`, which must be empty or not
    /// exist: the JPEGs under `images/`, a `.sfm-workspace.json`, and
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
    /// A photograph could not be encoded, decoded or written.
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
rendered, encoded and decoded once whichever of the two a caller needs.

**Why both the render and the decoded JPEG are kept.** Everything the value
takes from the photographs is computed from the decoded JPEGs, since those are
what a reader of the workspace sees
([The photographs are JPEGs](#the-photographs-are-jpegs)). The exact render is
kept beside them for the tests of the renderer, which compare a pixel with the
texture and need no JPEG error in the comparison.

**Why one options struct.** The scene's geometry (the plane, the cube, the
camera positions) is fixed, and is what the tests reason about; the options are
the parts a caller has a reason to change: a smaller image or fewer points for
a cheaper test, a larger bitmap side, no supersampling to test aliasing. It is
named `*Options` by [GLOSSARY.md](../GLOSSARY.md) § "Rust type names": the
caller fills it with choices and it has a `Default`. `for_point_count` is the
one setting *Load Demo Data* offers, a number of points, turned into the two
spacings: the points on a surface number about its area over the spacing
squared, so scaling both spacings by `sqrt(DEFAULT_POINT_COUNT / n)` scales
the count by about `n / DEFAULT_POINT_COUNT`. The count kept is near `n`, not
equal to it, since each grid is a whole number of rows and visibility drops
points.

**Why errors, not panics.** Options come from callers, and
[STANDARDS.md](../STANDARDS.md) § "Caller input never panics" applies: a zero
image size, a zero supersampling, a non-finite or non-positive spacing,
half-extent or focal length, a bitmap side of zero, or an obliquity outside
`0..=90` is `DemoSceneError::Options`. An option set that leaves some point
with fewer than two observations is not an error; that point is left out. An
option set under which a measurement endpoint lies in no observed patch is not
an error either; the measurement is left out
([The measurement](#the-measurement)).

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
// recon.point_set.points[i].normal is the surface's normal exactly, every
// keypoint is its point's projection rounded to f32, and the one distance
// measurement is a cube edge of 1 m.
```

And *Load Demo Data*, on a background thread:

```rust
let options = DemoSceneOptions::for_point_count(number_of_points)?;
let scene = DemoScene::render(&options, &progress)?;
let sfmr_path = scene.write_workspace(&temp_dir, &progress)?;
// then the same open as File > Open, of sfmr_path
```

## Python binding

The binding is `DemoScene` in `sfmtool.reconstruction`, the module that holds
`SfmrReconstruction`, in
[reconstruction/demo_scene.rs](../../crates/sfmtool-py/src/reconstruction/demo_scene.rs)
with its row in [python-bindings.md](../python-bindings.md) § "Submodules".
Its methods call the Rust ones and hold no logic of their own
([STANDARDS.md](../STANDARDS.md) § "Logic lives in Rust, not in Python
wrappers"), with the GIL released while they run.

```python
from sfmtool import ProgressCounter
from sfmtool.reconstruction import DemoScene

scene = DemoScene.render(
    *, image_width=640, image_height=480, focal_length_px=560.0,
    supersampling=2, ground_point_spacing=None, face_point_spacing=None,
    patch_half_extent=0.09, patch_bitmap_resolution=24,
    max_obliquity_deg=80.0, seed=0, point_count=None, progress=None,
)                                     # -> DemoScene
scene.image_names                     # list[str]
scene.rendered_photographs            # list of read-only (H, W, 3) uint8 views
scene.photographs                     # the decoded JPEGs, the same shape
scene.encoded_photographs             # list[bytes], the JPEG files
scene.reconstruction(progress=None)   # -> SfmrReconstruction
scene.write_workspace(dir, progress=None)  # -> pathlib.Path of demo.sfmr
```

The keywords are the fields of `DemoSceneOptions` with their defaults. The two
spacings default to `None`, which is their `Default` value, 0.25 and 0.125, so
that the binding can tell a spacing given from one left out: `point_count`
calls `for_point_count`, whose spacings then stand for the two, and cannot be
given with either. The other keywords apply on top of it. There is no
settings class: no binding exposes an `*Options` struct as a class, and the
options are keyword arguments, as the other bindings' settings are. The
photographs are read-only views of the scene's own pixels, whose base is the
`DemoScene`, so a read copies nothing and an array keeps its scene alive; the class is frozen and has no method that changes the scene.
`progress`, an optional `ProgressCounter`, is set to the count of the phase
that is running, as `upgrade_workspace`'s is: rows rendered and then images
encoded for `render`, points sampled and then bitmaps rendered for
`reconstruction`, and those followed by files written for `write_workspace`;
[STANDARDS.md](../STANDARDS.md) § "Long-running functions take a Progress"
asks a binding of a long call to take one. At the end of a call it holds its
last phase's count: 5 images, the point count (one bitmap per point), or 7
files (five JPEGs, the marker and the `.sfmr`). An option out of range, a
`point_count` of 0 or given with a spacing, or a `dir` that is not a
directory, holds files or lies inside a workspace raises `ValueError` with the
Rust error's message; a failed encode or decode, or a failed write of a
photograph, a directory, the marker or the `.sfmr`, raises `OSError` (the
subclass for its `io::ErrorKind` for a directory or the marker). Python's
signal handlers run while a call goes, as in `upgrade_workspace`: Ctrl+C
cancels the call's `Progress` and raises `KeyboardInterrupt`, and a cancel by
other means raises `InterruptedError`.

**Cost in a test session.** Rendering and building the value take well under a
second ([Cost](#cost)): through the binding, on the 32-thread x86-64 Windows
machine, `render` takes about 0.1 s, `reconstruction` 0.1 to 0.2 s and
`write_workspace` about 0.15 s. A fixture that renders once and writes one
workspace costs less than one solve of seoul_bull. The binding's tests share
one render, one value and one written workspace through module-scoped
fixtures, since only their module uses them; a later test module that needs
the scene moves them to a session-scoped fixture in `conftest.py`.

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
- **Background.** A ray that meets neither surface sees one flat colour, the
  dark slate `(60, 70, 80)`. No point lies on it.
- **No lighting.** A surface's colour is its texture alone, the same from
  every direction, so a surface point has the same colour in every photograph
  that sees it, up to the renderer's sampling, the rounding to 8 bits and the
  JPEG compression. This is what makes the patch bitmaps of one point agree
  between views, and it is the assumption every photometric score in the
  toolkit makes.

### Texture

Each surface carries fractional Brownian motion (fBm) of Perlin gradient
noise: five layers, or octaves, of the noise at lattice spacings from 0.32 m
down to 0.02 m, summed with falling amplitudes, so that the texture has light
and dark patches about a third of a face across as well as detail at the
scale of a patch, no
shift of it resembles another, and a patch can be located by its content.
The ground and each cube face use a different seed, so the cube does not blend
into the ground, and two faces meeting at an edge do not continue each other's
pattern: a surface's seed is `7 · seed + i`, where `seed` is the options' and
`i` the surface's index in the order ground, top, `-a`, `-b`, `+a`, `+b`,
base. The noise is evaluated in each surface's own 2D coordinates: `(x, y)`
on the ground, and on a cube face the coordinates along its two in-face axes
from one corner, each in `[0, 1]` on the face. A side face with outward normal
`n` takes `ẑ × n` and `ẑ`, the top `a` and `b`, and the base `a` and `-b`, the
same axes as the patch frames ([The reconstruction's columns](#the-reconstructions-columns)),
so each pair's cross product is the face's outward normal; the corner is the
one where both coordinates are 0.

**The noise functions** are in
[demo_scene/texture.rs](../../crates/sfmtool-core/src/reconstruction/data/demo_scene/texture.rs).
`perlin_noise(x, y, spacing, seed)` puts a unit gradient at each node of a
square lattice `spacing` apart, one of the eight unit vectors 45° apart,
picked by a hash of the node and `seed`; between nodes it blends the four
corners' linear ramps `g · (p − node)` with Perlin's quintic fade
`6t⁵ − 15t⁴ + 10t³`. It is 0 at every node, continuous with continuous first
and second derivatives, and its magnitude is at most `1/√2`
(`PERLIN_BOUND`), the value at a cell's centre when all four gradients point
at it. `perlin_fbm(x, y, &FbmOptions, seed)` sums `octaves` octaves of it:
octave `k` has the lattice spacing `spacing / lacunarityᵏ` and the amplitude
`gainᵏ`, and the sum is divided by the sum of the amplitudes, so its
magnitude is at most `1/√2` as well. Octave `k` takes its own seed, a hash of
`seed` and `k`, and its lattice is shifted by the fractional part of
`k · 0.618034` of its own cell along both axes. Without the seeds, every
octave would draw the same gradients at the same node indices, a pattern
repeated at each scale about the origin; without the shift, with a lacunarity
of 2 every node of a coarse octave would be a node of every finer one, where
all of them are 0. Both are computed in `f64` from `+`, `−`, `×`, `÷` and
`floor`, and the gradients are constants written without `sin` or `cos`, so
the same arguments give the same bits on every platform.

**The octaves.** The texture's fBm, `TEXTURE_FBM`, has 5 octaves, a coarsest
spacing of 0.32 m, a lacunarity of 2 and a gain of 0.8: lattice spacings of
0.32, 0.16, 0.08, 0.04 and 0.02 m, at amplitudes 1, 0.8, 0.64, 0.51 and 0.41.
The coarsest octave puts about three light or dark patches across a 1 m face,
each about 45 px across in a view 4 m away. The finest has the spacing 0.02 m.
A surface about 4 m from a camera with a focal length of 560 px is about 7 mm
per pixel, so a cell of the finest octave spans about 2.8 px in a view that
faces it: detail at the scale of a patch, as the bench tests choose it (their
plane texture is about 2.5 px a cell in a facing view,
[grown_plane_normal.rs](../../crates/sfmtool-core/src/bench/tests/grown_plane_normal.rs)).
The far ground and the ground at a grazing angle have cells under 2 px across,
which the supersampling averages rather than aliases.

**Contrast at the scale of a patch.** A patch is 0.18 m across and its bitmap
24 px; the photometric kernels the later steps run compare patches between views
by their content, which is the texture's variation within a patch. An fBm
spends part of the grey range on its coarse octaves, which change little
across a patch, and the octaves' sum is near its extremes at few points (its
RMS over the scene is about a quarter of its largest magnitude), so the
texture has less contrast within a patch than single-scale value noise at
0.02 m mapped onto the same grey range. Measured on the default scene's
renders, over each 24 × 24 px window that lies wholly on one surface: the
standard deviation of the grey, and the pixel-scale contrast, the RMS
difference between neighbouring pixels over `√2`, which is what the finest
octaves give. The values are the 10th percentile and the median over the
windows, in grey levels:

| Surface (windows) | Value noise, std. dev. | fBm, std. dev. | Value noise, pixel-scale | fBm, pixel-scale | fBm without the 0.02 m octave, pixel-scale |
|---|---|---|---|---|---|
| Ground (1347) | 23.6, 28.8 | 10.0, 13.2 | 11.6, 16.5 | 3.1, 4.3 | 2.3, 3.7 |
| Top (12) | 27.4, 29.0 | 11.6, 13.6 | 18.1, 18.7 | 4.6, 4.9 | 4.3, 4.7 |
| Face `-a` (45) | 27.6, 29.4 | 10.4, 12.4 | 11.0, 13.1 | 2.9, 3.3 | 2.2, 2.6 |
| Face `-b` (59) | 26.9, 28.0 | 9.0, 10.6 | 10.8, 12.6 | 3.0, 3.3 | 2.2, 2.5 |

The fBm keeps about 45% of the value noise's standard deviation within a
patch and about a quarter of its pixel-scale contrast. The gain is what
trades the coarse patches against the fine detail. Measured on the texture
itself, the medians of the standard deviation over 0.18 m windows on the four
surfaces are 9.5 to 11.2 levels at a gain of 0.5, 11.8 to 13.5 at 0.8 and 13.4
to 14.6 at 1.0; at 0.5 the texture looks blurred at the scale of a patch, and
at 1.0 the coarse patches fade into the fine grain, so 0.8 is used. Starting
the octaves at 0.64 m instead of 0.32 m lowers each of those by about a
quarter. The JPEG's error, a mean of 0.85 to 1.32 levels against the render
([The photographs are JPEGs](#the-photographs-are-jpegs)), stays well under
the pixel-scale contrast. A test asserts that the 10th percentile of each
seen surface's windows has a standard deviation of at least 8 levels and a
pixel-scale contrast of at least 2.6; without the 0.02 m octave the
pixel-scale contrast's 10th percentile falls to 2.2 on the two side faces and
2.3 on the ground, and the test fails.

**Normalisation.** The fBm's magnitude stays well under its bound, since the
octaves seldom reach their extremes at one point: its largest over every
surface of the scenes of seeds 0 to 15, sampled every 2 cm on the ground and
every 2 mm on the faces, is 0.462, 0.654 of `1/√2`. The texture maps the
magnitude `TEXTURE_PEAK`, 0.665 of `1/√2` or 0.470, to the ends of its grey
range, so the measured fBm takes up 98% of that range. A channel would clip
only where the fBm's magnitude passed 0.83 of `1/√2`, where the factor below
would fall under 0 or the largest tint channel, 250, rise over 255; that
needs the five octaves near their own extremes at one point, and no sampled
point comes near it.

The bench tests' noise is value noise, `value_noise(x, y, spacing, seed)`,
beside the Perlin noise in
[demo_scene/texture.rs](../../crates/sfmtool-core/src/reconstruction/data/demo_scene/texture.rs)
and built only for tests: pseudo-random levels on a square lattice from the
same hash of the node and the seed, blended between nodes with the smoothstep
`3t² − 2t³`, a value in `[-1, 1]` that each caller maps to its own range. It
was two copies in the bench tests, and both now call it: `noise` in
[grown_plane_normal.rs](../../crates/sfmtool-core/src/bench/tests/grown_plane_normal.rs)
as `128 + amplitude · value_noise`, the same bits as its copy gave, and
`other_texture` in
[grid_plane_normal.rs](../../crates/sfmtool-core/src/bench/tests/grid_plane_normal.rs)
as `127.5 + 97.5 · value_noise(x, y, 0.12, 0)`. Its copy had no seed term,
which is the seeded hash at seed 0, since the seed enters by an XOR with
`seed · constant`, and blended levels `30 + 195 · l`, which is
`127.5 + 97.5 · (2l − 1)`. The two forms are equal in exact arithmetic; in
`f64` they differ in the last bits at 47% of a million sampled points, by at
most `8.5 · 10⁻¹⁴`, which the bench tests' assertions, all tolerances, do not
see.

**Colour.** Each surface has its own tint, so that a person looking at the
viewer can tell from a patch's colour which face it is on. A surface's colour
is its tint scaled by the factor `0.5 + 0.4 · fbm / TEXTURE_PEAK`, one
factor for all three channels, in `[0.1, 0.9]` wherever the fBm's magnitude
is under `TEXTURE_PEAK`:

| Surface | Tint (RGB) | Hue | BT.709 luma of the tint |
|---|---|---|---|
| Ground | `(215, 180, 130)` | sand | 183.8 |
| Top, `+z` | `(150, 200, 150)` | green | 185.8 |
| Face `-a` | `(250, 160, 150)` | red | 178.4 |
| Face `-b` | `(150, 180, 250)` | blue | 178.7 |
| Face `+a` | `(235, 160, 235)` | magenta | 181.4 |
| Face `+b` | `(120, 195, 200)` | cyan | 179.4 |

The kernels read the photographs through `ImageToGray::default()`, BT.709 luma
unrounded, which is linear in the three channels, so a surface's grey is the
luma of its tint times that factor: the same noise as a grey scene,
at a gain set by the tint's luma. The tints are chosen with lumas within 5% of
each other, near 180, so every surface's grey runs over at most about `18`
to `167` and the noise has the same contrast on every surface; what the kernels read is
in substance the grey scene's. The largest channel times `0.9` is 225, so no
channel clips. The base, `-z`, is never seen and takes the ground's tint.

## The cameras

One shared pinhole camera, 640 × 480 px, focal length 560 px on both axes and
the principal point at the centre `(320, 240)`: a horizontal field of view of
about 59°. It has no distortion, no image has a per-image adjustment, and there
is no rig.

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

**Determinism.** The render depends only on the options. The texture's gradients
come from the hash of the lattice node and the seed, with no random number
generator, and the renderer uses `+`, `−`, `×`, `÷`, `sqrt` and `floor`, so a
rendered photograph is the same bytes on every platform. Rows are rendered in
parallel with rayon, each row written by one thread and summed in a fixed
order, so the thread count does not change a pixel.

### Cost

Five 640 × 480 photographs at four rays a pixel are 6.1 million rays.
A ray is one slab test, one plane test and one evaluation of the texture (five
octaves of Perlin noise, each four hashes and a blend), on the order of 50 to
100 ns, so 0.3 to 0.6 s on one thread and well under 0.1 s across 16
threads. Encoding and decoding five
640 × 480 JPEGs is a small part of that. The bitmaps are cheaper still: a
whole-cloud render of about 1,000 points takes 0.1 to 0.2 s on 32 threads
([reference-view.md § Cost of the stored bitmap](../core/patch/reference-view.md#cost-of-the-stored-bitmap)).
The step plan measures each.

Measured in the test build (optimization level 1) on a 32-thread x86-64
Windows machine, three runs each, while the machine was lightly loaded: the five
photographs render in 0.41 to 0.44 s on one thread and 0.030 to 0.035 s across
the thread pool; encoding the five JPEGs one after another takes 14 to 15 ms
and decoding them 10 ms; `DemoScene::render` as a whole takes 0.055 to
0.059 s. The same runs with single-scale value noise as the texture rendered
in 0.36 to 0.37 s on one thread, so the five octaves cost about 15% more; its
JPEGs, with more fine detail, were larger and took 23 to 24 ms to encode and
17 to 19 ms to decode. While another build loaded the machine the same runs
took up to twice as long on one thread and up to seven times as long on the
pool.

## The photographs are JPEGs

Each rendered photograph is encoded as a JPEG, and the value is built from the
JPEG as decoded, not from the render: the thumbnails, the image readings and
the patch bitmaps are computed from `ImageU8::decode_rgb` of the JPEG's bytes,
and `image_file_hashes` is the XXH128 of those bytes, which `write_workspace`
writes unchanged. Everything the scene fixes stays exact from the scene: the
poses, the points, the normals, the patch frames, the keypoints and the
measurement's endpoints. The JPEG moves no edge or texel, so a keypoint is
still where the photograph shows its point; it changes only the pixel values a
little, as a camera's own compression does.

**The in-memory value and the written file are the same reconstruction.** The
photograph-derived columns are computed from the pixels that decoding the
written file gives, so a save of the value needs nothing from disk: its
readings are its own rows (rule 1 of
[Where a writer gets the values](../formats/sfmr-file-format.md#where-a-writer-gets-the-values)),
and a load of the written `.sfmr` gives the value back, equal column for
column, with the workspace directory filled in. Dropping the bitmaps and adding
them again from the workspace's photographs gives the same bitmaps, on the
machine that built the value
([Determinism of the JPEG encoder and decoder](#determinism-of-the-jpeg-encoder-and-decoder)).

**Quality 95.** The JPEGs are written at `DEFAULT_JPEG_QUALITY`, 95, the
quality a caller of `ImageU8::write` passes when it has no reason to choose
another, the default of `sfmtool.fileio.write_image_rgb` and of OpenCV's
`cv2.imwrite` ([image.rs](../../crates/sfmtool-core/src/camera/image.rs)), with the
4:2:0 chroma subsampling `ImageU8::write` uses, so the demo's photographs are
compressed as the photographs the toolkit writes elsewhere are. Measured on a
full 640 × 480 frame of sand-tinted value noise at 2.8 px a cell with no
supersampling, which has more fine detail than the scene's texture, the decoded pixels differ from the encoded ones by a mean of
about 3 levels and at most about 25, and the file is 0.19 MB; the same frame
as PNG is 0.70 MB. The difference is the small one the maintainer asked for: a
patch's bitmaps no longer agree between views to the last level, as they do
not in a real capture.

**Size.** A rendered photograph has the blur of the supersampling and a flat
background, both of which compress better than that frame, so each JPEG is
under 0.2 MB and the five are under 1 MB: measured, 0.066 to 0.068 MB for
each low camera's and 0.097 to 0.098 MB for each high camera's, 0.40 MB in
all. The `.sfmr`'s largest column is the
patch bitmaps, 513 × 24 × 24 × 4 bytes, 1.2 MB before compression; the
whole default `.sfmr` is 0.75 MB.

### Determinism of the JPEG encoder and decoder

**The encoder gives the same bytes on every run of one path, and different
bytes on its two paths.** `ImageU8::encode` encodes a JPEG with the
`jpeg-encoder` crate (0.7.1,
[sfmtool-core's Cargo.toml](../../crates/sfmtool-core/Cargo.toml)), not with
the `image` crate's encoder, baseline, 4:2:0 with averaged chroma and the
optimized Huffman tables off. The crate has no floating-point arithmetic: its
RGB to YCbCr conversion is a fixed-point formula scaled by 2¹⁶, and its forward
DCT is the integer slow DCT ported from libjpeg's `jfdctint`. With its `simd`
feature, which sfmtool-core turns on, it chooses an AVX2 path at run time on an
x86 CPU that has AVX2 and the scalar path elsewhere, such as the macOS arm64 CI
runner. The AVX2 colour conversion gives the same integers as the scalar one:
the crate tests it (`avx_matches_scalar_rgb`), and an encode that converts the
colours with the scalar formula and runs the AVX2 DCT gives the AVX2 path's
bytes. The AVX2 DCT, a port of libjpeg-turbo's `jfdctint-avx2`, does not give
the scalar DCT's coefficients for every block. With the crate's `simd` feature
turned off, the same x86 machine encodes the five default photographs to other
bytes: decoded, 2.5% to 4% of their samples differ from the AVX2 files', by at
most 5 levels, the files differ in size by at most 27 bytes, and each path's
files are as close to the render as the other's, a mean error of 0.85 to 1.32
levels.

Step 1's test stores both paths' hashes of the five JPEGs and checks them
against the set of the path the encoder takes: the AVX2 set on CI's Linux and
Windows runners, and on the macOS arm64 runner the scalar set, computed on x86
with `simd` off, which so checks that the scalar path gives the same bytes on
x86 and arm64.

**Decided by the maintainer (2026-11-22): `simd` stays on.** A written demo
workspace's JPEGs, their `image_file_hashes`, and every column computed from
the decoded photographs differ between an x86 machine with AVX2 and any other
machine. Turning off `jpeg-encoder`'s `simd` feature would make the demo's
JPEG bytes the same everywhere, at the cost of encode time for every JPEG the
toolkit writes (on the five default photographs of the scene's first,
value-noise texture, 30 to 39 ms against 26 to 28 ms). Instead the encoder keeps its AVX2 path, and the tests store the file
hashes and the photograph-derived values once for each encoder path and check
the set of the path the encoder takes, as step 1's hash test does.

**The decoder can differ between platforms in the last level.**
`ImageU8::decode_rgb` decodes a JPEG with the `jpeg-decoder` crate (0.3.2,
default features off). That crate chooses SSSE3 code for its inverse DCT and
colour conversion at run time on x86, and scalar code elsewhere, and its
changelog says the SSSE3 output is not bit-for-bit the scalar output; its
`platform_independent` feature runs the scalar code everywhere, and the
workspace does not turn it on. So the photograph-derived columns (thumbnails,
readings, bitmaps) built on an x86 machine can differ from those built on an
arm64 one by a rounding level from the same JPEG bytes, on top of the
encoder's difference above, while every column the scene fixes is the same.
On any one machine the value and a load of the written file agree exactly, as
above. The workspace keeps `platform_independent` off, so the toolkit's
decoding of real photographs keeps its x86 speed: the demo's tests pin the
JPEG bytes of each encoder path exactly and compare the photograph-derived
columns to within one level against the values stored for the encoder path
taken.

**What the code has.** `ImageU8::encode(format, jpeg_quality) ->
Result<Vec<u8>, ImageError>` encodes to bytes in memory, with the checks of
`ImageU8::write`, which calls it, so both encode the same bytes; the demo
hashes and decodes the bytes before anything is written.
`ImageU8::decode_rgb(bytes)` decodes from memory with the same decoders as
`read_rgb`
([reading-and-writing-images.md](../core/camera/reading-and-writing-images.md)).

## The points and who sees them

**Sampling.** Points lie on a grid on each surface, each moved from its grid
position by a seeded offset of up to a quarter of the spacing along each axis,
so the points do not line up into rows that a regular pattern in the
photographs would echo:

- On the ground, `m = round(5 / ground_point_spacing)` rows and columns over
  `[-2.5, 2.5] × [-2.5, 2.5]`, at `−2.5 + (i + 0.5) · 5 / m` m along each
  axis, leaving out the points that land inside the cube's footprint: at
  0.25 m, `m = 20`, a grid of 400 of which 321 are observed by two cameras or
  more. The rest lie inside the footprint, near the cameras below the views'
  lower edges, or at the grid's corners outside the outer views.
- On each cube face but the base, `k = round(1 / face_point_spacing)` rows and
  columns at `(i + 0.5) / k` m from one corner along each in-face axis, so the
  first row is half a spacing from the edge: at 0.125 m, `k = 8` and 64 points
  a face. Only the top and the faces `-a` and `-b` keep points, 192: every
  point of the top is observed by the two high cameras, of `-a` by three and
  of `-b` by four. The four points nearest a face's corners are not moved, so
  that each cube corner lies inside a patch ([The measurement](#the-measurement)).

Both grids put each point at the centre of its cell, so the count on a
surface is close to its area over the spacing squared and `for_point_count`
scales it in proportion; a ground grid through the origin, `2n + 1` points a
side, would not, since its extra row and column are a larger share of a coarse
grid than of a fine one. 513 points at the defaults
(`DEFAULT_POINT_COUNT`). With a patch half-extent of 0.09 m, the
patches of the outer row on each face cross the cube's edge, and those of the
ground points next to the cube run into its side: those are the crease cases.
The ground points behind the cube, seen past it by the outer cameras and hidden
from the middle ones, are the occlusion cases.

**Visibility.** Camera `i` observes point `X` on a surface with normal `n` when
all of these hold:

1. the point's whole patch square lies in the photograph: each of its four
   corners, `X ± h·u ± h·v` with `u` and `v` the surface's in-face axes of its
   patch frame and `h` the patch half-extent, is in front of the camera
   (camera-space `z < 0`) and projects inside the image,
   `[0, 640) × [0, 480)`. An observation whose square runs off the image could
   not supply a full bitmap, and a point whose observations all did would
   have no reference view; at the defaults this rule leaves out 11 ground
   points near the image edges that the centre alone would keep, and takes
   one or two edge observations off 39 others;
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
points. Each column, and why it is exact or what it is computed from:

| Column | Value | Exact because |
|---|---|---|
| `cameras` | the one pinhole above | stated |
| `images/camera_indexes` | `0` for every image | |
| `images/quaternions_wxyz`, `translations_xyz` | each camera's world-to-camera pose | built from the stated centres and target; `f64` |
| `images/names` | `images/demo_0.jpg` … `images/demo_4.jpg` | |
| `images/image_file_hashes` | XXH128 of each JPEG's bytes | the bytes `write_workspace` writes |
| `images/thumbnails_y_x_rgb` | each decoded JPEG through `thumbnail_from_rgb` | the format's area resize of the photograph a reader decodes |
| `images/sharpness`, `gradient_structure_tensors_xx_yy_xy`, `image_to_gray` | `image_readings_from_rgb` of each decoded JPEG under `ImageToGray::default()` | computed from the photograph as decoded, as a writer that gathers them from the image file would ([sfmr-file-format.md § Image readings](../formats/sfmr-file-format.md#image-readings), rule 3) |
| `points3d/positions_xyzw` | the sampled points, `w = 1` | sampled on the surface; `f64` |
| `points3d/colors_rgb` | the texture's colour at the point | evaluated, not estimated |
| `points3d/reprojection_errors` | each point's RMS reprojection error, scored by `recompute_point_errors` | what the stored keypoints give: about `10⁻⁵` px, the rounding of the keypoints to `f32` |
| normals (`has_normals`) | `+z` on the ground, the face's outward normal on the cube | derived from the patch frame, `normalize(u × v)` in `f32`, as the format defines it ([sfmr-file-format.md § Derived normals](../formats/sfmr-file-format.md#derived-normals)) |
| `points3d/patch_u_halfvec_xyz`, `patch_v_halfvec_xyz` | a square patch frame of half-extent 0.09 m in the surface: on the ground `u = h·x̂`, `v = h·ŷ`; on a side face with normal `n`, `v = h·ẑ` and `u = h·(ẑ × n)`; on the top `u = h·a`, `v = h·b` | each makes `u × v` point along the surface's outward normal, toward the cameras that see it |
| `tracks/image_indexes`, `point_indexes`, `observation_counts` | the observations visibility keeps, sorted by `(point, image)` | the ray–box test is exact |
| `tracks/keypoints_xy` | each point's projection into the image, `f32` | projected through the camera; the keypoint is where the photograph sees the patch's centre, since the photograph was rendered from the same pose |
| `tracks/reference_observations` | the reference-view rule's pick for each point | the same render `--add-patch-bitmaps` runs |
| `points3d/patch_bitmaps_y_x_rgba` | each point's reference observation's `24 × 24` render | rendered from the decoded JPEGs by `render_patch_cloud_bitmaps` with `UnreferencedPoints::Pick` |
| `measurements/distance/` | one measurement, a cube edge of 1 m ([The measurement](#the-measurement)) | the edge's length is the cube's side; the endpoints are the corners' projections |
| `derived/` depth statistics | `recompute_depth_statistics` | computed from the poses and points |
| `metadata` | `operation: "demo"`, `world_space_unit: "m"`, `feature_source: "embedded_patches"`, `distance_measurement_count: 1` | |

Absent: rigs and frames, per-image adjustments, normal confidence, point
constraints, observation confidence and observation readings.

**The bitmaps come from the photographs, not from the texture.** The format
says a point's bitmap is the render of its reference observation from that
photograph, so that dropping the bitmaps and adding them again gives the same
bitmaps ([sfmr-file-format.md § `tracks/reference_observations`](../formats/sfmr-file-format.md#tracksreference_observationsnint32zst-with-the-patch-frame)).
Sampling the texture over the patch instead would give a sharper bitmap that no
later render reproduces. The generator renders them through the same function
the viewer's display render and `--add-patch-bitmaps` reach
([stored_bitmap.rs](../../crates/sfmtool-core/src/patch/stored_bitmap.rs)),
with the decoded JPEGs in memory as `TiledPhotograph`s. The texture is then a
check on the renderer: a test can compare a facing point's bitmap with the
texture sampled over its patch, to within the supersampling's blur and the
JPEG's error.

**Two things the render settles that the geometry does not.** The bitmaps are
the reference-view rule's renders at the stored keypoints, with
`KeypointSubpixelParams::default()` at the options' bitmap side, as
`--add-patch-bitmaps` renders them. Every point gets a pick, since each of its
observations' patch squares lies in its photograph (visibility rule 1). The
rule's pick depends on the pixels: it compares the views by their
content, and for a point whose two best views score nearly alike a change of
one level in the photographs changes the pick, and with it the whole bitmap.
On the default scene one point of the 513, point 348 on the top, picks the
other view under the scalar JPEG encoder. Before visibility rule 1 took the
whole patch square, one point of the top also picked the other view under
`jpeg-decoder`'s scalar path (its `platform_independent` feature) on the same
JPEG bytes; the top is observed only by the two high cameras. So
a machine of another encoder or decoder path can store a few bitmaps of
another view. Decided by the maintainer (2026-11-22): the generator keeps the
production reference-view rule rather than a pick made from the geometry, so
its bitmaps are the ones `--add-patch-bitmaps` stores, and the tests tolerate
the flip: they compare the bitmaps within a level only where the pick agrees,
and allow one sampled point whose pick differs ([step plan](#step-plan),
step 2). Neither measured flip is among the sampled points, so the allowance
covers a path not measured, such as arm64's decoder.

### The measurement

The value holds one distance measurement in the version 17 `measurements/`
section ([sfmr-file-format.md § 10. Measurements](../formats/sfmr-file-format.md#10-measurements-optional-version-17)):
the cube's vertical edge where the faces `-a` and `-b` meet, which is exactly
1 m long. Its corners are `-0.5·a - 0.5·b` at the base and at the top,
`(-0.1, -0.7, 0)` and `(-0.1, -0.7, 1)`. Both faces are observed by the
cameras at −27.5° and 0° ([The cameras](#the-cameras)), and each corner is in
view of all five cameras. The measurement fills the distance kind's columns:

| Column | Value |
|---|---|
| `distances` | `[1.0]` |
| `units` | `[2]`, `"m"` in the canonical legend |
| `metadata.json` | `distance_measurement_count: 1` and the canonical `unit_names` `["mm", "cm", "m", "in", "ft"]` |
| `image_indexes` | `[[2, 1]]`: the base corner in `demo_2` (0°, 1.2 m) and the top corner in `demo_1` (−27.5°, 3.0 m) |
| `endpoints_xy` | each corner's projection into its image, rounded to `f32`, in the convention of `tracks/keypoints_xy` |
| `labels.json` | `["cube edge"]` |

The top-level `distance_measurement_count` is 1, and the content hash gains
`measurements_xxh128`. In Rust this is `ImageSet::measurements`, a
`MeasurementsData` whose `distance` is a `DistanceMeasurementsData`. The two
endpoints are in different images on purpose: an endpoint resolved through one
view carries none of that view's keypoint residual into the distance
([measurement-endpoint.md § From the pixel to a texel](../core/reconstruction/measurement-endpoint.md#from-the-pixel-to-a-texel)),
and two images test that.

**The endpoints resolve to texels of observed patches.** An endpoint is a
pixel, and `resolve_measurement_endpoint` gives it a 3D position only when the
square of an observation in its image holds the pixel
([measurement-endpoint.md](../core/reconstruction/measurement-endpoint.md)).
A corner is not a sampled point, so it must lie inside the patch of one that
is. On each face the point nearest the corner is not moved from its grid
position, half a spacing, 0.0625 m, from both edges, and the corner is at
texel `(±0.694, ±0.694)` of its 0.09 m patch, inside `[−1, 1]²`. Each corner
lies in such a patch on each of the faces `-a` and `-b`, and the top corner on
the top as well, and the points of `-a` and `-b` are observed in both endpoint
images. Every candidate square at a corner's pixel that is not behind the
front layer lies in a plane through the corner: a face that meets at the
corner, the top, or the ground at the base. So whichever candidate is picked,
the texel names the corner itself: the endpoint's position is
`X_p + s·u + t·v` on that point's own patch, which lies in the surface's
plane. The resolved `scene_distance` is 1 m to within about `10⁻⁶` m, the
rounding of the keypoints and endpoints to `f32` at about 7 mm a pixel. On the
default scene the endpoints are `(303.803, 339.316)` in `demo_2` and
`(353.795, 229.849)` in `demo_1`. The base corner's pixel lies in the squares
of the corner points of `-a` and `-b`, and the rule picks `-b`'s, at texel
`(−0.694, −0.694)`; the top corner's lies in those of the top, `-a` and `-b`,
and the rule picks `-a`'s, at `(0.694, 0.694)`. The resolved `scene_distance`
is `1 + 2.2 · 10⁻⁷` m. The
corners stay covered while half the face spacing is below the patch
half-extent, `k ≥ 6`; when an option set leaves an endpoint in no square, the
generator leaves the measurement out, as it leaves out a point with too few
observations.

## The workspace

`write_workspace(dir)` writes, into an empty or missing `dir`:

```
dir/
├── .sfm-workspace.json
├── demo.sfmr
└── images/
    ├── demo_0.jpg
    ├── …
    └── demo_4.jpg
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

**The marker's writer.** `write_sfmtool_workspace_marker(dir, options)` in
[workspace/marker.rs](../../crates/sfmtool-core/src/workspace/marker.rs) writes
the version 2 marker for the `sfmtool` tool, from `SfmtoolWorkspaceOptions`
(the feature cap and the grey conversion `sfm ws init` offers, with
`Default`): `SiftParams::default()` with `double_image_below_long_side` at
4096, where the Rust default has no limit, the `feature_type`, and the
`feature_prefix_dir` named from them. Python's `init_workspace`
([_workspace.py](../../src/sfmtool/_workspace.py)) calls it through its
binding for the `sfmtool` tool, and `get_default_sfmtool_feature_options`
returns its options, so the two cannot drift. It writes the text Python's
`json.dump(record, f, indent=2)` wrote, keys in the same order, with LF line
breaks; the Python writer, opening the file in text mode, wrote CRLF on
Windows. `write_workspace` calls it with the default options. The workspace
marker's file name is defined three times in Rust (`WORKSPACE_MARKER` in
`sfmtool-archive-io`, `WORKSPACE_CONFIG_FILE` in `features::sift::files` and
`WORKSPACE_FILE_NAME` in `workspace`); the writer uses the first.

## Load Demo Data

Today *File > Load Demo Data...* opens a dialog asking for a number of points
and appends a node holding `SfmrReconstruction::demo(n)` with no path
(`AppState::load_demo` in [ops.rs](../../crates/sfm-explorer/src/state/ops.rs),
the dialog in [modals.rs](../../crates/sfm-explorer/src/app/modals.rs)). With
this proposal the dialog keeps its *Number of points* box, which now sets the
scene's point spacings through `DemoSceneOptions::for_point_count`. The box
starts at `DEFAULT_POINT_COUNT`, 513, and takes 300 to 10,000: below
about 250 the face spacing passes twice the patch half-extent and the
measurement's corners are no longer covered, and 10,000 points keep the bitmap
render to about a second. The count kept is near the number asked for, and the
Action Log line names it. *Load* then:

1. starts a background task, as *File > Open* does, that renders the scene,
   creates a new directory for it in the system temporary directory
   (`std::env::temp_dir()`), and calls `write_workspace` into it, reporting
   through the task's `Progress`;
2. opens the written `demo.sfmr` by the same open as *File > Open*
   (`AppState::start_open` in
   [open.rs](../../crates/sfm-explorer/src/state/open.rs)), so the node is
   labelled `demo` from the file's stem and its workspace, photographs and
   photograph cache are those of any opened file;
3. deletes the directory when the viewer exits.

**One directory per load, and how a crashed viewer's directory is removed.**
Each load writes its own directory, `sfmtool-demo-<process id>-<n>`, with `n`
counting the loads of this viewer; a second load in one session does not reuse
the first's directory, since the first node may still be open. Before it
creates the directory, the viewer creates a lock file beside it in the
temporary directory, `sfmtool-demo-<process id>-<n>.lock`, and holds an
exclusive lock on it (`std::fs::File::try_lock`, stable since Rust 1.89, below
the workspace's MSRV of 1.98) for as long as it runs. The operating system
releases a lock when its process ends, however it ends. At exit the viewer
deletes each of its directories, then releases and deletes the lock file. At
start the viewer looks for `sfmtool-demo-*.lock` files in the temporary
directory and, for each one whose lock it can take, deletes the directory of
the same name and the lock file: their owner has ended. A
`sfmtool-demo-*` directory with no lock file beside it has no owner either,
since an owner creates the lock file first and deletes it last, and is deleted
too. The process ID in the name only keeps two running viewers' names apart; it
does not decide whether the owner is alive, since the operating system reuses
process IDs. Nothing in the viewer or core does this today: no Rust code checks
whether another process is alive or takes a file lock, so the lock file and
the sweep at start are new.

A lock file exists unlocked for a moment between its creation and the
`try_lock`, and a sweep in another viewer that starts in that moment could take
the lock and delete the file. Step 4 closes that gap, for example by having the
sweep delete only a lock file whose directory exists and leave a lone lock file
to a later sweep once it is older than the sweep's start, and by having a
viewer that fails its own `try_lock` move on to the next `n`.

The node now has a path, so several things the viewer and its specs say of
demo data change:

- `path` is the file in the temporary workspace rather than `null` on the wire
  ([mcp-server.md](../gui/mcp-server.md)).
- The demo node saves as any other node with a file does
  ([saving.md](../gui/saving.md)): *Save* writes its `demo.sfmr` in the
  temporary workspace, and *Save As* writes where the user chooses. A file
  saved inside the temporary workspace is deleted with it when the viewer
  exits. A file saved elsewhere records the path to the temporary workspace,
  as any *Save As* records its workspace, and once that directory is deleted
  it opens without photographs, as an `embedded_patches` file with thumbnails
  and bitmaps can.
- `--demo`, which `ui_basic` uses to set a viewer up
  ([architecture.md § Testing](../gui/architecture.md#testing)), becomes a
  background open that a test waits on, as it waits on a file named on the
  command line.
- The dialog's *Number of points* box, which the MCP `type_text` test types
  250 into, takes 300 to 10,000 and starts at `DEFAULT_POINT_COUNT` rather
  than 1000, so that test types a number in the new range.

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

The Python tests that could use the binding are those that want a known
answer on a small workspace with photographs, rather than a solve of
seoul_bull or kerry_park:
[test_measurement_endpoint_rust_bindings.py](../../tests/rust_bindings/reconstruction/test_measurement_endpoint_rust_bindings.py)
and [test_scale_by_measurements.py](../../tests/xform/test_scale_by_measurements.py),
which would have a measurement of exactly 1 m to resolve and scale by;
[test_sfmr_image_readings.py](../../tests/test_sfmr_image_readings.py), whose
readings can be checked against the photographs the workspace holds; and
[test_web_export.py](../../tests/test_web_export.py) and
[test_render_patches.py](../../tests/patch/test_render_patches.py), which need
patch frames, bitmaps and photographs.

The tests that only want a cheap value with no pixels keep the sphere demo: the
3D view's upload tests, the reconstruction data and per-image adjustment tests,
the infinity conversion, alignment and reprojection-noise tests, and most of
the viewer's scene graph and save tests. Each migration is decided test by
test, by whether the test's assertions gain from real pixels; a test that gains
nothing from them stays on the sphere demo.

## Step plan

Each step is one change on its own branch, with its tests.

1. **Renderer, scene and JPEGs.** The shared value noise moved into the
   library, with the two bench tests calling it; the Rust writer of a version
   2 `sfmtool` workspace marker, which Python's `init_workspace` calls;
   `ImageU8::encode` to bytes, split out of `ImageU8::write`;
   `DemoSceneOptions` with `for_point_count`, and `DemoScene::render` with the
   tinted texture, a Perlin-noise fBm, and the JPEG encode and decode. Tests:
   the Perlin noise is 0 at its lattice nodes, continuous in value and slope
   across a lattice line, reaches its bound `1/√2` at a cell's centre and
   never passes it, changes with its seed, and gives stored bits; the fBm
   adds each octave at its spacing, shift, seed and amplitude and stays
   within the bound; the texture's largest fBm magnitude over four scenes is
   under `TEXTURE_PEAK` and within 10% of it; on each surface the photographs
   show, the 10th percentile of the 24 × 24 px windows has a grey standard
   deviation of at least 8 levels and a pixel-scale contrast of at least 2.6
   ([Texture](#texture)); the rendered
   photographs are the same bytes on two runs and across thread counts; the
   five JPEGs match XXH128 hashes stored in the test for each encoder path,
   checked on every CI platform, so that the x86 AVX2 encoder path and the
   arm64 scalar one are each checked; `photographs()` is `decode_rgb` of
   `encoded_photographs()`; a facing pixel's colour in the render is the
   texture at the point its centre ray hits, to within the supersampling's
   blur; each surface's grey is its tint's luma times the noise factor; the
   cube's silhouette is where the slab test puts it. Measure the render's and
   the JPEG encode and decode's time.

   *Built.* The generator is
   [demo_scene.rs](../../crates/sfmtool-core/src/reconstruction/data/demo_scene.rs)
   with `texture` and `render` in `demo_scene/`, re-exported from
   `sfmtool_core::reconstruction`. It has `render`, `options`,
   `image_names`, `rendered_photographs`, `photographs` and
   `encoded_photographs`; `tiled_photographs` comes with step 2, which is
   the first to read it, and so do `DemoSceneError`'s `Directory`, `Save`
   and `Io`. `DEFAULT_POINT_COUNT` was 540, this draft's estimate, until step
   2's test of the sampling fixed it. `render` checks every option, the
   sampling's among them, and also refuses an image side over 65535, the
   most a JPEG holds. The JPEG hash test stores a set of hashes for each
   encoder path, since the two paths give different bytes
   ([Determinism of the JPEG encoder and decoder](#determinism-of-the-jpeg-encoder-and-decoder)).
   The facing-pixel test also renders with `supersampling` 1, where the
   pixel is the texture at its centre ray's hit up to the rounding, since
   the four-sample comparison alone cannot see a texture misplaced by a
   fraction of a pixel. Measured times are in [Cost](#cost). The texture
   was single-scale value noise at first and is now the Perlin fBm the
   maintainer asked for (2026-11-22); the value noise stays beside it for
   the bench tests.
2. **The reconstruction.** Point sampling, visibility, the columns, the
   measurement, the bitmaps; `DemoScene::reconstruction` and
   `DemoScene::write_workspace`. Tests: the defaults keep
   `DEFAULT_POINT_COUNT` points, and `for_point_count` keeps within a few
   percent of the count asked for; every keypoint is its point's projection;
   no observation's segment meets the cube; a ground point behind the cube is
   hidden from the middle camera and seen from an outer one; the derived
   normals are the surface normals; the measurement's endpoints are the
   corners' projections, each resolves with `resolve_measurement_endpoint` to
   a texel of an observed patch at its corner, and
   `resolve_distance_measurements` gives a `scene_distance` of 1 m to within
   `10⁻⁶` m; an option set that leaves a corner in no patch leaves the
   measurement out; the value saves, verifies and loads back equal, with the
   workspace directory set; the image file hashes match the written files; the
   thumbnails, readings and bitmaps are those computed from the written files
   decoded again; the photograph-derived columns match the values stored for
   the encoder path taken, within a tolerance of one level, on every CI
   platform.

   *Built.* The points and visibility are
   [demo_scene/points.rs](../../crates/sfmtool-core/src/reconstruction/data/demo_scene/points.rs)
   and the columns, the measurement and `tiled_photographs`
   [demo_scene/value.rs](../../crates/sfmtool-core/src/reconstruction/data/demo_scene/value.rs);
   `write_workspace` and the `Directory`, `Save` and `Io` errors are in
   [demo_scene.rs](../../crates/sfmtool-core/src/reconstruction/data/demo_scene.rs).
   `DEFAULT_POINT_COUNT` is 513: 321 on the ground and 64 on each of the top,
   `-a` and `-b`, with 1,712 observations. Two changes from the sampling this
   draft first described. Visibility rule 1 takes the whole patch square
   rather than the point's centre: with the centre alone, a ground point at
   `(0.083, −2.160, 0)` was kept with its keypoint 0.2 px from the lower edge
   of `demo_2` and its square only 73% inside `demo_1`, the reference-view
   rule took neither view, and the point was stored with no reference and an
   empty bitmap, which the viewer draws as a ball rather than a surfel. Of
   the 524 points the centre rule kept, the square rule leaves out 11 ground
   points, that one among them, and no point is left without a reference.
   And
   the ground grid puts each point at the centre of its cell, as the faces'
   grids do, rather than on a grid through the origin with `2n + 1` points a
   side, whose count did not scale with the inverse square of the spacing
   ([The points and who sees them](#the-points-and-who-sees-them)); with the
   grid through the origin, `for_point_count(10_000)` kept 9,464 points.
   Measured with both changes, `for_point_count` keeps 290 for 300, 994
   for 1,000, 1,993 for 2,000, 2,935 for 3,000 and 9,751 for 10,000. Since
   a grid side is a whole number, neighbouring counts can give the same
   grids: over every tenth count from 300 to 10,000 the largest miss is at
   350, which keeps 380, 8.6% over, and every miss from 2,300 up is under
   3%. Below
   300 the misses grow (92 for 100). The test checks 300, 1,000, 3,000 and
   10,000 to within 5%, by the sampling alone, with no render.

   `render_patch_cloud_bitmaps` renders the bitmaps with the reference-view
   rule; every point gets a pick, and the rule's pick changes for one point
   under the other encoder path
   ([Two things the render settles](#the-reconstructions-columns)). The
   per-path values the last test stores are a digest rather than the
   columns: three thumbnail texels of each image, each image's four readings,
   and one texel and the reference of every 36th point's bitmap. Texels
   compare within one level. The readings compare within 1% (`Exy`, which is
   near 0, within 1% of `√(Exx · Eyy)`): a one-level change at every sample
   would bound them only loosely, about half of `√Exx`, while the decoder's
   scalar and SSSE3 paths, a one-level difference at some samples, gave
   readings within `6.6 · 10⁻⁴` of each other and the two encoder paths
   within `3 · 10⁻⁴`. The two paths' sampled references and thumbnail texels
   are the same, three sampled bitmap texels differ by a level, and their
   readings differ in the fourth digit. A sampled point whose
   pick differs from the stored one is skipped, and at most one may.

   **The patch bitmaps' contrast.** Over the stored 24 × 24 bitmaps of the
   default scene, with the grey and the two contrasts of
   [Texture](#texture), the 10th percentile and median by surface, in grey
   levels: the ground (321 bitmaps) 9.2 and 11.9 standard deviation, 2.22
   and 2.77 pixel-scale; the top 8.7 and 10.8, 2.37 and 2.58; `-a` 10.1 and
   12.4, 3.12 and 3.37; `-b` 9.0 and 10.9, 2.91 and 3.19. The least
   textured bitmap has a standard deviation of 6.5 levels. These are near
   the renders' windows (a standard deviation of 9 to 11.6 at the 10th
   percentile), with the pixel-scale contrast of the ground and the top
   somewhat lower, since a bitmap resamples its patch from a grazing or
   oblique view. Every bitmap holds texture of several levels against the
   JPEG's error of about one, so the texture was kept as it is; whether it is
   enough for the photometric kernels is for the steps that run them on the
   scene to show.

   Measured in the test build on the 32-thread x86-64 Windows machine,
   while the other tests of the module ran: `reconstruction` takes 0.10 to
   0.15 s and `write_workspace`, which builds the value again, 0.12 to
   0.15 s. The step 2 tests, with the scene's render, finish in 0.5 to 0.9 s.
3. **Python binding.** `DemoScene` in `sfmtool.reconstruction`, its row in
   [python-bindings.md](../python-bindings.md) and its entry in
   [test_reconstruction_patches_registration.py](../../tests/rust_bindings/reconstruction/test_reconstruction_patches_registration.py),
   the `reconstruction` submodule's registration test. Tests: the binding's value
   equals the Rust one's on the same options; `write_workspace` writes a
   workspace that `sfm inspect` reads; `ProgressCounter` climbs; each error
   raises the stated exception.

   *Built.* The binding is
   [reconstruction/demo_scene.rs](../../crates/sfmtool-py/src/reconstruction/demo_scene.rs),
   tested by
   [test_demo_scene_rust_bindings.py](../../tests/rust_bindings/reconstruction/test_demo_scene_rust_bindings.py).
   The binding's value is checked against the Rust one by the values the Rust
   tests store: the five JPEGs' XXH128 equal one of the two encoder paths'
   stored sets, the defaults keep 513 points and 1,712 observations, and
   `point_count=1000` keeps 994, the count step 2 measured for
   `for_point_count(1000)`, here on 64 × 48 photographs with the default's
   field of view. Building it settled the
   [Python binding](#python-binding) section's details: the spacings default to `None` so that `point_count` can
   refuse one given beside it, the options are keyword arguments rather than a
   settings object, the photographs are views of the scene's pixels, the
   counter holds the running phase's count, and the exceptions are those the
   section lists, with `InterruptedError` and Ctrl+C added. The migration of
   the Python tests listed above, which this step first held, moved to step 5,
   so that this step adds the binding alone. The binding's tests take 0.7 s
   on one worker.
4. **Load Demo Data.** The menu entry and `--demo` load the scene from a
   temporary workspace, with the *Number of points* box mapped through
   `for_point_count`; the lock file, the deletion at exit and the sweep at
   start; the gui specs ([architecture.md](../gui/architecture.md),
   [mcp-server.md](../gui/mcp-server.md), [saving.md](../gui/saving.md),
   [user-experience.md](../gui/user-experience.md)) and the `ui_basic` and MCP
   tests follow. Tests: a directory whose lock file no process holds is
   removed at start, and one whose lock a live process holds is kept. This
   step changes what the viewer shows, so its PR carries screenshots.
5. **Test migration.** The Rust tests and the Python tests listed above, one
   area at a time, each kept only where real pixels make an assertion possible
   or exact.
