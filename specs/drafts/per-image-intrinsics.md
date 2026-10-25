# Per-Image Intrinsic Adjustments

**Status:** Draft. Decided:
- the bundle adjustment first releases each camera's shared principal point, with a checkbox in SfM Explorer's Bundle Adjust dialog and an argument on its MCP tool (PR 1);
- an `.sfmr` file stores, per image, a small adjustment of its camera's focal and principal point, as one optional `images/` column per adjustable parameter, with each camera recording in `cameras/metadata.json` which parameters vary per image; this is format version 13 (PR 2);
- the focal adjustment is a relative scale, `f_i = f·(1 + s_i)`, and the principal-point adjustment is per axis as a fraction of the image size, `cx_i = cx + a_x,i·width` and `cy_i = cy + a_y,i·height`, so that both carry over unchanged when a reconstruction moves from low-resolution proxies to the full-resolution photographs;
- one accessor, `image_intrinsics(i)`, returns an image's camera with its adjustment applied, borrowed when there is none, and nothing else reads the raw columns (PR 2); every consumer that looks up an image's camera goes through it, and COLMAP export writes each camera that has a non-zero adjustment on any of its images as one COLMAP camera per image, with the adjustment applied (PR 3);
- the bundle adjustment releases the adjustments per image, each held toward zero by a Gaussian prior whose width is a fraction of the image (PR 4);
- one camera entry per image is rejected as the way to do this.

Not decided: the default widths of the per-image priors (provisionally 2% for each, to be measured on more datasets, Kerry Park among them); a focal model driven by the EXIF subject distance. See [Open questions](#open-questions).

Amends:
- [core/geometry/bundle-adjustment.md](../core/geometry/bundle-adjustment.md): the kernel's lens block, which releases no principal point, and its image block, which holds the pose alone
- [core/reconstruction/bundle-adjust.md](../core/reconstruction/bundle-adjust.md): `CameraRelease`, the report and the write-back
- [gui/edits/bundle-adjust.md](../gui/edits/bundle-adjust.md): the dialog's camera rows, the version label and the Action Log entry
- [formats/sfmr-file-format.md](../formats/sfmr-file-format.md): the images and cameras sections and the version history

## Purpose

A reconstruction stores a small list of cameras, and each image says which of them took it. Every image taken through one camera is then read through exactly the same lens: the same focal length, the same principal point (the pixel the optical axis passes through) and the same distortion. For a phone camera that is not quite true. Optical image stabilisation moves the lens sideways by a few pixels from one photograph to the next, which moves the principal point, and autofocus changes the focal length slightly with the focus distance ("focus breathing"). A bundle adjustment that must read every photograph through one shared lens cannot model either effect, and leaves it in the residuals as an error that grows toward the edge of the image.

This draft lets each image carry a small adjustment of its camera's focal length and principal point, stored in the `.sfmr` file beside the pose, applied wherever an image's camera is read, and solved by the bundle adjustment under a prior that keeps it small. The shared camera keeps the distortion, which is the part the images have in common and the part that needs the most data to fit. It is laid out as four pull requests, each of which leaves the tree working.

## Evidence

The measurements below were made by an experiment outside the tree, a joint bundle adjustment of poses, points and intrinsics with a Huber loss and nothing trimmed. Each is a **held-out** median reprojection error: the solve is fitted on half the points; the other half are then triangulated with the poses and intrinsics fixed, and their reprojection error is measured. A parameter that only fits noise lowers the fitted half's error and not the held-out half's.

**`dino_dog_toy`** (85 photographs, a Pixel 7 Pro main camera, 6.81 mm f/1.85 with optical image stabilisation, 2040×1536, EXIF subject distance 0.15 to 0.45 m). Its candidate ground truth has one shared `SFMTOOL_PINHOLE` camera, and its residuals grow radially toward the image edge in a way no single shared camera fits.

| Released | Held-out median |
|----------|-----------------|
| shared focal and radial distortion (baseline) | 0.818 px |
| + shared principal point, tangential, thin prism, aspect and skew | 0.757 px |
| + per-image focal | 0.732 px |
| + per-image principal point | 0.322 px |
| + per-image focal and principal point | 0.280 px |

- The per-image principal-point offsets spread over about 21 px in x (1.0% of the width) and 20 px in y (1.3% of the height), about a shared offset of roughly (−7, +19) px. They drift along the capture sequence and correlate only weakly with the direction of gravity in the camera frame, which fits the stabilised lens's position at the moment of capture.
- The per-image focal scale spreads over 1.3 to 1.6%, and correlates at −0.89 with the EXIF `SubjectDistance`: focus breathing.
- Without a prior, one image with 63 observations moved its principal point by −626 px. That is why the per-image release in PR 4 has a prior.

**`seoul_bull_sculpture`** (17 images, 270×480, `SIMPLE_RADIAL`, its checked-in ground truth) shows the same effect at a smaller size: held-out median 0.233 px at the baseline, 0.199 px with a shared principal point (a shared x offset of about −10.6 px), 0.142 px with a per-image principal point. The per-image offsets spread over about 3.2 px in x (1.2% of the width) and 3.7 px in y (0.8% of the height), and the focal scale over 0.5%.

**Kerry Park** (two fisheyes on a rig) was not measured.

The experiment is not reproducible in the tree yet: the dino ground truth is not checked in, and the kernel cannot release any of these parameters. PR 4's tests use synthetic scenes, and a script that repeats the held-out measurement on the checked-in ground truths is listed in [Open questions](#open-questions).

## The adjustment

An image `i` taken through camera `c` (width `w`, height `h`, focal `f`, or `fx` and `fy` on a two-focal model, principal point `(cx, cy)`) is read through the camera's model and distortion with

```
f_i  = f  · (1 + s_i)          (fx and fy both, on a two-focal model)
cx_i = cx + a_x,i · w
cy_i = cy + a_y,i · h
```

and nothing else changed. `s_i`, `a_x,i` and `a_y,i` are the image's **intrinsic adjustment**. All three are `0` for an image whose camera does not vary them, and an image whose three are `0` is read through the shared camera exactly.

**Why a relative focal scale.** The scale is dimensionless, so it means the same at any resolution: a camera resized by a factor `k` has its focal multiplied by `k`, and `f_i` follows with `s_i` unchanged. A proxy workflow can therefore solve on low-resolution copies of the photographs, switch to the originals by resizing the shared camera, and keep every adjustment as it is. `(1 + s)` was chosen over `exp(s)` because the stored number then reads directly as "this image's focal is 1.3% longer", and because over the 1 to 2% range the evidence shows the two differ by less than `s²/2`, 0.02% at most. The one thing `exp(s)` gives that `(1 + s)` does not, a focal that cannot reach zero, is a step guard here (`s > −1`, and the image's focal above `1e-6` like the shared one). On a two-focal model the one scale moves `fx` and `fy` together, which keeps the aspect ratio the shared camera holds.

**Why a fraction of the image size, per axis.** A resize maps a pixel coordinate as `x' = x·w'/w` and `y' = y·h'/h`, exactly under the format's pixel convention (origin at the top-left corner of the image, half-pixel centres; [sfmr-file-format.md](../formats/sfmr-file-format.md) § "Pixel space"). An offset stored as a fraction of the width in x and of the height in y is therefore unchanged by any resize, including one that changes the aspect ratio. It is not stored as a fraction of the focal, because the focal can itself be released, per camera and per image, and an offset that changed in pixels whenever the focal moved would couple two parameters the solve should keep apart.

**The distortion centre moves with the principal point.** In every model here the distortion is applied in normalised coordinates before the focal and principal point are, so the distortion centre is the principal point. A shift of the principal point, shared or per image, moves the distortion centre with it. That is what the evidence measures, and it is what a stabilised lens does.

### Rejected: one camera entry per image

A reconstruction can already give every image its own camera entry. That would also let each image's focal and principal point differ, but each image would then carry its own unshared copy of the distortion, and a bundle adjustment releasing the distortion would solve one spline per image, each from that image's observations alone. The distortion is the part of the lens the images do share, and the part that needs observations from many images to reach the edge of the field. The adjustment keeps it shared and lets only the three parameters that do vary move per image.

## PR 1: releasing the shared principal point

The bundle adjustment kernel ([geometry/bundle_adjust.rs](../../crates/sfmtool-core/src/geometry/bundle_adjust.rs)) releases a camera's focal and its lens distortion and holds its principal point. PR 1 lets it release the principal point too, per camera.

**The release.** `CameraRelease` gains `principal_point: bool` beside `focal` and `distortion`. The kernel's `bundle_adjust` gains an `opt_pp: bool` request after `opt_bspline`, narrowed per camera by `BaCameras::releases` as the others are. Unlike the focal, the principal point is released on every model: in every model `u = fx·x_d + cx` with `x_d` independent of `cx`, so the columns `∂(u, v)/∂(cx, cy)` are the identity, exact at every incidence angle. It needs no focal release beside it either, since it is not a scale; `DistortionWithoutFocal` stays the only pairing rule.

**The reduced camera system.** Each camera's lens block gains two slots, `cx` and `cy`, after its spline slots, so `F_SLOT`, `K1_SLOT` and `BSPLINE_SLOT0` keep their values and a camera's block is `[f, k1, c₀..c_{N−1}, cx, cy]`, `4 + N_j` wide. The per-observation camera block grows by the same two columns (`BASE_CAM_COLS` becomes `4 + 6`, `BSPLINE_CAM_COLS` becomes `4 + BSPLINE_SUPPORT + 6`). A slot that is not released is pinned with an identity row and column and a zero gradient, as the focal and `k1` slots are today, so its column carries exact zeros. A solve that releases no principal point is meant to give the same output bit for bit as before PR 1; the PR checks that by running the kernel's and the reconstruction adjustment's test scenes before and after and comparing every output, and reports the comparison.

`Lens` gains the current `cx` and `cy`, and `Lens::at` builds the candidate camera with them through a new crate-private `CameraIntrinsics::with_principal_point`. The `k1` step guard measures the imaged field (`field_r`) about the current principal point, as it does today about the fixed one.

**A step guard.** A candidate step whose principal point is not finite, or lies outside the image (`cx` outside `[0, width]` or `cy` outside `[0, height]`), is rejected whole and re-damped, as a non-finite focal is. No physical lens has its optical axis off the sensor, and a solve that tries to put it there is following a degenerate direction, not the data.

**The near-degeneracy, and what PR 1 does about it.** A principal-point shift and a small rotation of every image's camera move the image almost alike: in a pinhole camera a shift of `δ` pixels in `cx` and a rotation of `δ/f` about the camera's y axis differ only by a term of order `δ·x²/f²` at image coordinate `x`, plus whatever the distortion adds, since the distortion centre moves with the principal point and not with the rotation. So the shared principal point is determined by the field's width and by the distortion. With a wide field or strong distortion it is well determined; with a narrow field and weak distortion the cost barely changes along that direction.

PR 1 adds no prior on the shared principal point. It is one parameter pair fitted from every image of the camera. In `seoul_bull_sculpture` releasing it alone improved the held-out error (0.233 → 0.199 px, a shared x offset of about 10.6 px); in `dino_dog_toy` it was released together with the other extra global terms, which improved it (0.818 → 0.757 px), and the shared offset found there was about 20 px. In a weak direction the kernel's Levenberg-Marquardt damping, which scales each diagonal entry by `1 + λ`, already keeps the step short, and the step guard keeps it on the sensor. The kernel computes no covariance today (its spec lists covariance estimation as a non-goal), and the reduced system it solves is singular along the gauge, so reporting a conditioning number for the principal point needs a pseudo-inverse over the gauge's null space; that is left to [Open questions](#open-questions). What PR 1 reports instead is the shift itself, so a person sees a principal point that moved far.

**The surfaces.**
- `CameraAdjustment` gains `principal_point_released: bool`, `principal_point_before: [f64; 2]` and `principal_point_after: [f64; 2]`; the write-back replaces the camera's principal point with the solved one where it was released.
- Python: `EditedReconstruction.bundle_adjust(..., opt_pp=False, ...)`, a `"principal_point"` key in each `releases` entry (the error for an unknown key names the three keys), and the report's camera dicts carry the three fields. The kernel binding `sfmtool.geometry.bundle_adjust` gains `opt_pp=False`.
- SfM Explorer: each camera row of the Bundle Adjust dialog ([bundle_adjust_prompt.rs](../../crates/sfm-explorer/src/bundle_adjust_prompt.rs)) gains a third checkbox, **Release principal point**, clear by default and enabled on every model. The version label names the released parts of each camera in the order focal, principal point, lens distortion (`, focal and principal point released`; `, focal, principal point and lens distortion released`), and the Action Log entry appends each released principal point's change, `, principal point (1020.0, 768.0) → (1013.1, 787.2)`, with the camera index over several cameras as the focal's change has. This is a visible change, so the PR carries before and after screenshots of the dialog.
- MCP `bundle_adjust` gains `release_principal_point`, a default for every camera like `release_focal`, and the same optional field on each `cameras` entry ([gui/mcp-server.md](../gui/mcp-server.md) § "`resect_camera_image` / `bundle_adjust`").
- `sfm xform --bundle-adjust`: `BundleAdjustTransform.release_principal_point` ([_bundle_adjust.py](../../src/sfmtool/xform/_bundle_adjust.py)), which today reaches only the pycolmap path and which the `--bundle-adjust` parser always leaves `False` (`sfm densify` passes its own `release_principal_point` to pycolmap and does not build the transform), is passed to the sfmtool path as well, and its docstring stops saying sfmtool's adjustment always holds the principal point. The `--bundle-adjust` value gains no key in PR 1.
- The glossary's **release** entry adds the principal point.

**Tests.** In the kernel: the `cx`, `cy` columns against a central difference of the projection on every model; on a distortion-free `SIMPLE_PINHOLE` scene with a field of about 60°, a principal point planted 2% of the width off centre recovered from the centre with noise-free observations; on the same scene narrowed to a field of about 5°, the solve converging with the principal point inside the image and the residual median no higher than with it held, which pins the degeneracy's behaviour without asserting a recovery the data cannot support; a candidate off the sensor rejected. In the reconstruction adjustment: the release written back and reported, a held principal point unchanged, and two cameras releasing each its own. In the viewer: the checkbox's answer reaching the release list, the label and the Action Log line; `ui_basic` for the dialog's rows.

## PR 2: the format columns and the accessor

PR 2 makes the adjustment representable everywhere a reconstruction is held, and readable through one function, while no writer can yet produce a non-zero one.

**The `.sfmr` format, version 13.** The images section gains two optional columns:

| Entry | Shape | Holds |
|-------|-------|-------|
| `images/focal_adjustments.{N}.float64.zst` | `(N,)` | `s_i`: the image's focal is its camera's times `1 + s_i` |
| `images/principal_point_adjustments_xy.{N}.2.float64.zst` | `(N, 2)` | `[a_x,i, a_y,i]`: the image's principal point is its camera's plus `a_x,i·width` in x and `a_y,i·height` in y |

Each camera in `cameras/metadata.json` gains an optional key `per_image_adjustments`, a list naming the parameters that vary per image for that camera, `"focal"` and `"principal_point"`, in that order. A camera that varies neither has no key, so the cameras JSON of a file without adjustments is byte for byte what version 12 writes. The key is what tells a zero adjustment that was solved for apart from one that was never enabled.

The rules a reader checks, refusing the file otherwise:
- a column is present exactly when at least one camera lists its parameter;
- a row whose image's camera does not list the parameter holds exactly `0` (a writer writes `+0.0`);
- every value is finite, and every focal adjustment is above `−1`;
- `per_image_adjustments` names only `"focal"` and `"principal_point"`, each at most once;
- a file below version 13 has neither the columns nor the key.

`float64`, the type of the poses beside them, so that a solved adjustment round-trips exactly. Both columns are part of `images_xxh128` in their lexicographic slots: `focal_adjustments` after `feature_tool_hashes` (or after `camera_indexes` in an `embedded_patches` file) and before `image_file_hashes` and `metadata.json`; `principal_point_adjustments_xy` after `names.json` and before `quaternions_wxyz`. The camera key is part of `cameras_xxh128`, which hashes the cameras JSON. `SFMR_FORMAT_VERSION` becomes 13, with a `SFMR_PER_IMAGE_ADJUSTMENTS_VERSION = 13` beside `SFMR_REFERENCE_OBSERVATIONS_VERSION` ([types.rs](../../crates/sfmtool-sfmr-format/src/types.rs)). Migration is mechanical: a version 12 file reads with every adjustment zero and no camera listing any parameter, and saves as version 13 with neither columns nor key; a version 13 file with no column maps back to version 12 unchanged apart from the version number.

In the format crate, `SfmrData` gains `focal_adjustments: Option<Array1<f64>>` and `principal_point_adjustments_xy: Option<Array2<f64>>`, and `SfmrCamera` gains `per_image_adjustments: Vec<String>`, skipped in the JSON when empty. The PR files the columns into [sfmr-file-format.md](../formats/sfmr-file-format.md) (images section, cameras section, content hash, a "Version 12 → Version 13" section and the history line), in the present tense.

**In memory.** The adjustment travels with its image, so `SfmrImage` gains `intrinsic_adjustment: IntrinsicAdjustment`, a `Copy` struct `{ focal: f64, principal_point: [f64; 2] }` whose `Default` is all zero. Every edit that reorders, filters, merges or copies images then carries it without a change of its own; the five non-test places that build an `SfmrImage` (the reader's conversion, the demo, `reconstruction/edit.rs` and two in the bindings' `clone.rs`) set it. The per-camera flags go on the image set beside `cameras`: `ImageSet::per_image_adjustments: Vec<PerImageAdjustments>`, one `PerImageAdjustments { focal: bool, principal_point: bool }` per camera, or empty when no camera varies anything, the way an empty `BundleAdjustOptions::releases` means every camera held. An edit that adds or removes cameras keeps it in step with `cameras`.

**The accessor.** One function applies an adjustment:

```rust
impl ImageSet {
    /// The intrinsics image `image_index` was taken with: its camera's, with the
    /// image's intrinsic adjustment applied. Borrowed when the adjustment is all
    /// zero, which is every image of a file without adjustments.
    pub fn image_intrinsics(&self, image_index: usize) -> Cow<'_, CameraIntrinsics>;
}

impl SfmrReconstruction {
    /// `self.image_set.image_intrinsics(image_index)`.
    pub fn image_intrinsics(&self, image_index: usize) -> Cow<'_, CameraIntrinsics>;
}

impl CameraIntrinsics {
    /// This camera with `adjustment` applied: every focal times `1 + focal`,
    /// the principal point moved by the fractions of the width and height.
    pub fn with_adjustment(&self, adjustment: &IntrinsicAdjustment) -> CameraIntrinsics;
}
```

It is named for what it returns, the image's intrinsics, as `refit_camera_intrinsics` is named for the intrinsics rather than the camera: a *camera* in this project also has a pose, and an *image* already means an entry of the image set addressed by index. `Cow` because almost every image of almost every file has no adjustment, and a consumer on a hot path (a projection per observation) should not clone a spline camera per call. Nothing else reads `intrinsic_adjustment` or the raw columns: a consumer that wants an image's lens calls `image_intrinsics`, and a consumer that wants the shared camera (the Camera Intrinsics panel, the camera model switch, the bundle adjustment's lens block) reads `cameras[camera_index]` and says so. `ImageSet::camera_for_image` keeps returning the shared camera until PR 3 moves its callers and removes it. Python gets `SfmrReconstruction.image_intrinsics(i)` and the two columns and the camera key in the `read_sfmr` / `write_sfmr` dict.

**The guard.** Until every consumer reads through the accessor, a non-zero adjustment in a file would be read wrongly by the ones that do not. So in PR 2 `read_sfmr` and `write_sfmr_with_options` refuse a file, or a value, with any non-zero adjustment, with an `SfmrError` that names the first such image and says this build does not yet read per-image intrinsic adjustments. All-zero columns and camera keys are accepted, so a version 13 file can carry the flags. The guard is at the file boundary only: a value built in memory may hold non-zero adjustments, which is how PR 3's tests check each consumer before the guard is lifted.

**Tests.** A round trip of both columns and the camera key, bit for bit; each reader refusal; a version 12 file reading with zero adjustments and saving as version 13 with no columns; the images and cameras hashes changing with the columns and the key; `image_intrinsics` borrowing on a zero adjustment and applying a non-zero one exactly, on a single-focal and a two-focal model; an image filter and an image reorder carrying the adjustment with its image; the guard refusing a non-zero row on read and on write.

## PR 3: every consumer reads the image's intrinsics

**The sites.** Counted on `main` at 8642723b with `grep -E "camera_indexes|\.camera_index\b|cameras\[|camera_for_image|image_camera"` over non-test `.rs` files, the tree has about 360 lines that touch an image's camera or the camera column: `sfmtool-core` 147, `sfmtool-py` 98, `sfmtool-sfmr-format` 46, `sfm-explorer` 44, `sfmtool-colmap` 16, `sfmtool-camrig-format` 12. Most of the format crates' and the bindings' lines move the `camera_indexes` column between layers and need no change. The lookups of one image's camera, `.camera_index` or `camera_for_image`, are about 85 lines in about 54 files: 46 in 26 files of `sfmtool-core` (11 files under `reconstruction/`, 7 under `analysis/`, 2 each under `bench/`, `geometry/` and `patch/`, `web_export.rs`, and the example `align_vs_congealing.rs`), 35 in 25 files of `sfm-explorer` (the scene renderer's frustums, background image and track rays, the MCP replies, state, Image Detail, Track View, the bench, camera lock and the intrinsics panels), and 4 in 3 files of `sfmtool-py`. On the Python side, 25 files under `src/sfmtool/` read `camera_indexes`, `camera_index` or index `cameras[...]`, most of them `camrig/cp.py`, `merge/reconstructions.py`, `colmap/io.py`, `_undistort_images.py`, `colmap/db_export.py` and `_densify.py`. Each site is either moved to `image_intrinsics` or marked as one that means the shared camera.

**The split.** Three PRs, each of which can merge while the guard holds, since with every adjustment zero the accessor returns the shared camera and nothing changes. Each one's tests build a value in memory with non-zero adjustments and check that the consumers it moved read them.

- **PR 3a, the library.** `sfmtool-core` and the bindings that look up an image's camera. This includes the bundle adjustment reading held adjustments: `BaCameras` gains `image_adjustments: Option<Cow<'a, [IntrinsicAdjustment]>>`, one per image, and every observation, the trim's in-front measure, the inter-round re-estimation and the noise level read the image's adjusted camera, so that an adjustment the solve does not release is honoured rather than dropped. Reprojection errors, the reprojection noise, triangulation, resection, Add Image to Tracks, the bench and the patch renders all read through the accessor. `switch_camera_model` refits the shared camera and keeps every adjustment as it is, since the adjustment is relative to whatever the shared camera is.
- **PR 3b, the viewer.** Every `sfm-explorer` site, including what the 3D view draws (frustums, background image, track rays), Image Detail and Track View. `get_camera_image` adds the image's `intrinsic_adjustment` and its adjusted focal and principal point to its reply. Whether Image Detail shows the adjustment, and where, is decided in that PR and carries screenshots.
- **PR 3c, Python and the exports, and lifting the guard.** The 25 Python files; COLMAP export; the guard removed from `read_sfmr` and `write_sfmr_with_options`.

**COLMAP export.** COLMAP has one camera per camera entry and no per-image intrinsics, so `save_colmap_binary` ([colmap/io.py](../../src/sfmtool/colmap/io.py)) and `write_colmap_binary` ([colmap_io/write.rs](../../crates/sfmtool-colmap/src/colmap_io/write.rs)) write a camera with a non-zero adjustment on any of its images as one COLMAP camera per image of it, each that image's `image_intrinsics` in the camera's own model, and every other camera once, as today. COLMAP rig sensors hold one camera each, so a reconstruction with rigs whose sensor camera would be split is refused, naming the camera, rather than written with a rig that no longer describes it. `create_colmap_db_from_reconstruction` ([colmap/db_export.py](../../src/sfmtool/colmap/db_export.py)) follows the same rule, so a COLMAP solve started from it begins at the adjusted intrinsics. Import is unchanged: a COLMAP reconstruction with one camera per image reads as one sfmtool camera per image, and nothing folds such cameras into adjustments.

**The other writers.** `sfm to-nerfstudio` writes each frame's adjusted intrinsics per frame where an adjustment is non-zero. `sfm camrig cp` copies the shared camera out of a `.sfmr`, since a `.camrig` holds no per-image data, and says so when the source has adjustments. Merging reconstructions (`merge/reconstructions.py`) carries each image's adjustment with it, and where it maps two cameras to one, does so only when their `per_image_adjustments` agree as well.

## PR 4: releasing the adjustments per image

**The release.** `CameraRelease` gains `image_focal: bool` and `image_principal_point: bool`: whether each image of the camera has its focal scale, and its principal-point offset, solved. The kernel takes them as `opt_image_f` and `opt_image_pp` requests narrowed by `BaCameras::releases`, like the others. The per-image principal point is released on every model, for the same reason as the shared one. The per-image focal is released on the models `focal_is_releasable` admits, and `ImageFocalNotReleasable` refuses it on any other, naming the camera. A per-image release does not need the shared one: with both released, the shared parameter takes the part every image has in common, because the prior pulls the mean adjustment toward zero; with the shared one held, the adjustments are solved about it.

**The reduced camera system.** Each image's block grows by the released per-image parameters. When any camera of the solve releases one, every image block is `[δθ×3, δt×3, s, a_x, a_y]`, nine wide, and the slots an image's camera does not release are pinned; when none does, it stays six wide, so a solve with no per-image release is the solve before PR 4 bit for bit. `LinState::img_slot` becomes `n_shared + IMG_COLS·ci`, and the per-observation camera block grows by three columns in the instantiations that carry them (with and without the spline, so four widths in all). The points are still eliminated by the Schur complement; the per-image parameters couple to them through the observations, as the poses do.

The columns, at the image's adjusted camera:

```
∂(u, v)/∂s_i   = ((u − cx_i), (v − cy_i)) / (1 + s_i)     (= f·(x_d, y_d))
∂(u, v)/∂a_x,i = (w, 0)
∂(u, v)/∂a_y,i = (0, h)
```

The shared focal's column becomes `(u − cx_i)/f`, the existing column read about the image's principal point, and the spline columns take the image's focal `f_i` in place of `f`.

**The prior.** Each released adjustment is held toward zero by a Gaussian prior. The principal point's is isotropic in pixels, with standard deviation `σ_pp = p_pp · max(w, h)`; the focal scale's is `σ_s = p_s`. With the data cost approximately `Σ r²` in pixels for residuals inside the loss scale, the prior enters the round's cost as

```
σ_n² · Σ_i ( ((a_x,i·w)² + (a_y,i·h)²) / σ_pp²  +  s_i² / σ_s² )
```

where `σ_n` is the reprojection noise level measured at the start of the round over its kept observations (the measure `round_noise` takes after the final round today), fixed for the round so that the cost the damping ladder compares is one function per round, as the spline penalty's `f_ref` is. The prior's rows are added to each image block after the data and before the damping, like the spline penalty's. The widths are settings, `BundleAdjustOptions::image_prior: ImagePriorOptions { focal_sigma, principal_point_sigma }`, both fractions, defaulting provisionally to `0.02`: on the evidence, two to three times the principal-point spread `dino_dog_toy` and `seoul_bull_sculpture` show (41 px against about 21 px on the dino's long side of 2040, 9.6 px against about 3.7 px on the bull's 480) and above the focal-scale spread of 0.5 to 1.6%, so a real adjustment is pulled only a little, while an image like the one that ran to −626 px is held. The defaults are fixed only after the measurement in [Open questions](#open-questions).

The prior is what makes the per-image release well posed. For one image a principal-point shift and a rotation of that image alone differ only by the field-curvature term of PR 1 and the distortion centre's motion, and a focal scale and a move of that camera along its optical axis differ only by the scene's depth variation; with few observations either difference is below the noise.

**Step guards.** A candidate step is rejected and re-damped when an image's focal `f·(1 + s_i)` is not finite and above `1e-6`, or its principal point is not finite or leaves the image, as for the shared parameters.

**Write-back and the report.** Each posed image of a camera with a per-image release takes its solved adjustment, and the camera's `per_image_adjustments` gains the released parameters, so that a solved zero is recorded as solved. An unposed image keeps the adjustment it had. Each `CameraAdjustment` gains `image_adjustments: Option<ImageAdjustmentSpread>`, present when the camera released a per-image parameter: the number of images, the mean and standard deviation of the focal scale (as fractions) and of the principal-point offset (in pixels per axis), and `near_prior: Vec<NearPrior>`, the images whose adjustment ended more than two prior standard deviations from zero, each with the parameter and its distance in standard deviations. An adjustment that far out is one the data pulled well past what the prior expects, so these are the images a person should look at. The Action Log entry adds one line per such camera, and the version label names the per-image releases (`, focal and principal point released per image`).

**The surfaces.** Python: `opt_image_f=False` and `opt_image_pp=False` on `EditedReconstruction.bundle_adjust` and on the kernel binding, `"image_focal"` and `"image_principal_point"` keys in `releases`, the two prior widths as keywords, and the spread in the report; the kernel binding takes and returns the `(N, 3)` adjustments. MCP: `release_image_focal` and `release_image_principal_point`, defaults and per-camera fields. The dialog: **Release focal length per image** and **Release principal point per image** on each camera row, with screenshots. `sfm xform --bundle-adjust` gains no key in PR 4.

**Tests.** On a synthetic scene with planted per-image principal-point offsets of about 1% of the width and focal scales of about 1%, the released solve recovering each within a tolerance set by the observation count; an image with few observations and a zero planted adjustment staying within its prior, where with a very wide prior it drifts; the column checks against central differences; parity with no per-image release; the write-back setting the camera's flags; the spread and the near-prior list.

## Later

- **A focal driven by the subject distance.** The per-image focal scale in `dino_dog_toy` follows the EXIF `SubjectDistance` closely. A camera could model it as `f_i = f₀ + a/D_i`, two shared parameters fitted from every image, in place of one free scale per image. That is a different parameterisation and is not part of this sequence.
- **The dino ground truth.** Once PR 4 lands, rebuild the `dino_dog_toy` ground truth with per-image adjustments and refit its spline, since the spline fitted today has absorbed part of what the adjustments model.

## Words

Glossary entries the PRs add when they land:

| Word | Means | PR |
|------|-------|----|
| **release** (extended) | adds the shared principal point (`CameraRelease::principal_point`, `opt_pp`, `release_principal_point`), and the per-image focal and principal point (`image_focal`, `image_principal_point`) | 1, 4 |
| **intrinsic adjustment** | an image's focal scale and principal-point offset from its camera's shared intrinsics; `IntrinsicAdjustment`, `SfmrImage::intrinsic_adjustment`, the `images/focal_adjustments` and `images/principal_point_adjustments_xy` columns. Not *per-image camera*, which would read as the rejected camera entry per image. The stored names say *adjustment* rather than *offset* because a zero adjustment then reads as "no adjustment" for both parameters | 2 |
| **image intrinsics** | the intrinsics an image was taken with, its camera's with its adjustment applied; `image_intrinsics(i)` | 2 |

## Open questions

- **The prior widths.** 2% for each is provisional. Measure the held-out error against the prior width on `seoul_bull_sculpture`, `dino_dog_toy` and Kerry Park (whose two fisheyes are the first case with a rig and a wide field), with a script under `scripts/` that repeats the held-out split, before fixing the defaults.
- **Conditioning of the shared principal point.** Whether the report should carry the solved principal point's standard deviation, read from the final linearization's reduced system with the gauge's null space projected out, so that a narrow-field solve can say its principal point is not determined rather than only how far it moved.
- **The CLI.** Whether `sfm xform --bundle-adjust` gains keys for the principal point and the per-image releases.
- **Two-focal models.** The focal-scale column `(u − cx_i, v − cy_i)/(1 + s_i)` is exact on a two-focal model as well, since one scale moves both focals. Whether PR 4 widens the per-image focal release past `focal_is_releasable` is open.
