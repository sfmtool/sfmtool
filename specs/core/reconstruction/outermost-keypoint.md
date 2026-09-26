# The outermost keypoint of a camera

A camera model describes the lens out to some angle from the optical axis, and
the spline models (`SFMTOOL_FISHEYE`, `SFMTOOL_PINHOLE`) make that angle an
explicit choice: the spline's domain end, past which the model is a straight
line. The one measurement that says how far out a camera's photographs actually
reach is the keypoint that lies furthest from the principal point. This spec
describes the function that finds it for each camera of a reconstruction, in two
populations: the reconstruction's own observations, and every feature detected
in the images' `.sift` files. On a circular fisheye the two differ: the
observations stop where tracks stop, and the detected features go on to the
edge of the image circle.

The number is shown wherever the spline domain is edited, so a person can set
the domain from it: the bundle adjustment's domain control
([`bundle-adjust.md`](bundle-adjust.md),
[`../../gui/edits/bundle-adjust.md`](../../gui/edits/bundle-adjust.md)), the
switch's report ([`switch-camera-model.md`](switch-camera-model.md)) and MCP
`get_camera_intrinsics` ([`../../gui/mcp-server.md`](../../gui/mcp-server.md)).
It does not change any default: the spline domain still defaults to the far
image corner ([`../camera/refit-camera-intrinsics.md`](../camera/refit-camera-intrinsics.md)).

## Rust API

The function lives in
[outermost_keypoint.rs](../../../crates/sfmtool-core/src/reconstruction/outermost_keypoint.rs),
as `sfmtool_core::reconstruction::outermost_keypoint`, and is bound as
`SfmrReconstruction.outermost_keypoints`.

```rust
pub struct KeypointReach {
    pub radius_px: f64, // distance from the camera's principal point
    pub theta_deg: f64, // incidence angle the camera's model gives the pixel
    pub image: usize,   // the image it is in, by table index
    pub xy: [f64; 2],
}

pub struct OutermostKeypoints {
    pub camera: usize,
    pub images: usize,                    // images that use the camera
    pub observed: Option<KeypointReach>,  // among the reconstruction's observations
    pub detected: Option<KeypointReach>,  // among every feature of the .sift files
    pub detected_images: usize,           // .sift files read for `detected`
}

pub fn outermost_keypoints(
    recon: &SfmrReconstruction,
    cameras: &[usize],
    read_sift_files: bool,
) -> Vec<OutermostKeypoints>; // one per camera, ascending
```

```rust
use sfmtool_core::reconstruction::outermost_keypoint::outermost_keypoints;

for camera in outermost_keypoints(&recon, &[0, 1], true) {
    if let Some(detected) = camera.detected {
        println!("camera {}: {:.1} px, {:.1}°", camera.camera, detected.radius_px, detected.theta_deg);
    }
}
```

### Why it is shaped this way

**Outermost by radius, reported with its angle.** The keypoint is chosen by its
pixel distance from the principal point, which is a property of the photograph
alone, and its angle is then read through the camera's model. A caller that has
just refitted or solved a camera measures under that camera, so the angle
follows the model the domain is being chosen for. The radius does not.

**Two populations, both reported.** The observations are always there, and are
what the reconstruction can actually constrain; the detected features say how
far the image content goes, which on a circular fisheye is the image circle.
The domain control offers the detected angle and falls back to the observed one,
and labels which one it shows, because the two can be ten degrees apart.

**Reading files is the caller's choice.** `read_sift_files` is a flag rather
than a side effect, because the reconstruction-level bundle adjustment reads
nothing off disk and reports only the observed keypoint, while the switch, the
dialog and the CLI read the files. An unreadable file is skipped and counted
out, never an error: a value that sits beside no `.sift` file (an
`embedded_patches` ground truth copied elsewhere) reports `detected: None` and
nothing else changes.

## What it measures

For each chosen camera, over every image that uses it, posed or not:

- **observed**: each observation of the image, at its inline keypoint where the
  value carries one, otherwise at the feature its `.sift` file holds (read only
  under `read_sift_files`);
- **detected**: every feature position of the image's `.sift` file, under
  `read_sift_files`.

A keypoint whose pixel is not finite is skipped. The angle is
`off_axis_angle_deg`
([report.rs](../../../crates/sfmtool-core/src/camera/report.rs)), the angle
between the optical axis and the ray the model gives the pixel.

On `kerry_park` tk107, switched to an eight-coefficient `SFMTOOL_FISHEYE`, the
first sensor's outermost observation is 230.3 px, 95.8°, and its outermost
detected feature 259.2 px, 108.8°; the second sensor's are 217.4 px, 89.2° and
255.2 px, 103.4°. Reading the positions of all 48 `.sift` files takes about
8 ms.

## Implementation notes

**Positions only.** Detection reads the positions entry of each `.sift` file and
nothing else (`read_sift_positions`), which is what keeps the whole rig to a few
milliseconds; the descriptors are never decompressed.

**One read per image serves both populations.** For a `sift_files` value the
observed pixel is the file's feature at the observation's index, taken from the
positions already read for the detected keypoint.

## Python bindings

`SfmrReconstruction.outermost_keypoints(*, cameras=None, read_sift_files=True)`
returns a list with one dict per camera: `camera`, `images`, `observed` and
`detected` (each `None` or a dict of `radius_px`, `theta_deg`, `image`, `xy`) and
`detected_images`.

## Testing

[outermost_keypoint/tests.rs](../../../crates/sfmtool-core/src/reconstruction/outermost_keypoint/tests.rs)
writes a `.sift` file per image of the demo reconstruction, every feature on a
line out from the principal point and one feature per image further out than any
observation uses: the detected keypoint is that one, the observed is the largest
observed index, and the angle is the model's for the pixel. Without readable
files (not asked to read, or moved away) nothing is detected; inline keypoints
are observed without reading a file; a camera past the table is ignored. The
switch and the bundle adjustment test their report entries in their own suites,
and the bindings in
`tests/rust_bindings/test_edited_reconstruction_rust_bindings.py`.
