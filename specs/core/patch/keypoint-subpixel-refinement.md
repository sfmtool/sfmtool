# Photometric Subpixel Keypoint Refinement

Photometric subpixel keypoint refinement takes the keypoints of one 3D point's
observations, already close to correct, and moves each to sub-pixel accuracy by
locally maximizing how well the image content around it matches the point's
reference render: the patch tile of the point's reference observation, rendered
at that observation's own keypoint. It samples each image at fractional
positions and follows its gradients. It does no global search and no view
selection, and it does not move the reference observation. Its typical caller
is [keypoint localization](patch-keypoint-localization.md), which places the
views against the same reference render on an integer grid first, but it is
usable on any approximately correct keypoint set.

This is the **high-accuracy** refinement: a continuous solve reaches the optimum
of the photometric objective, where a grid search is quantized to its grid. It
is the option to reach for when accuracy matters most.

## Rust API

The refiner is in
[keypoint_subpixel.rs](../../../crates/sfmtool-core/src/patch/keypoint_subpixel.rs),
with its parameters and result in
[keypoint_subpixel/params.rs](../../../crates/sfmtool-core/src/patch/keypoint_subpixel/params.rs).
It is bound as `PatchCloud.refine_keypoints` and run by the pipeline in
[_embed_patches.py](../../../src/sfmtool/_embed_patches.py).

```rust
pub fn refine_patch_keypoints(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],          // one per reconstruction image
    view_set: &[u32],                      // the views to refine
    starting_keypoints: Option<&[Option<[f64; 2]>]>, // parallel to view_set
    reference: Option<usize>,              // position in view_set of the reference
    params: &KeypointSubpixelParams,
) -> KeypointRefinement;

pub fn refine_patch_keypoints_reporting(/* the same */, progress: &Progress<'_>)
    -> KeypointRefinement;

pub fn refine_patch_cloud_keypoints(
    cloud: &PatchCloud,
    views: &[ProjectedImage<'_>],
    view_sets: &[Vec<u32>],
    starting_keypoints: Option<&[Vec<Option<[f64; 2]>>]>,
    references: Option<&[Option<usize>]>,  // parallel to the cloud
    params: &KeypointSubpixelParams,
    progress: &Progress<'_>,
) -> Result<Vec<KeypointRefinement>, Cancelled>;

pub enum ReferenceTemplate<'a> {
    Bitmap(&'a [u8]),                          // an R×R RGBA bitmap, as a .sfmr stores it
    Observation { image: u32, keypoint: [f64; 2] },
}

pub fn refine_view_against_reference(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    template: ReferenceTemplate<'_>,
    target: u32,
    target_keypoint: [f64; 2],
    params: &KeypointSubpixelParams,
) -> Option<([f64; 2], f64)>;              // the keypoint and its ECC score

pub fn fuse_patch_bitmap(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    view_set: &[u32],
    keypoints: &[[f64; 2]],
    params: &KeypointSubpixelParams,
) -> Option<Vec<u8>>;
```

**Why it is shaped this way.**

- **The reference is an argument**, as in the localizer: the caller knows the
  point's stored reference observation, or the reference a bench track holds,
  and passes its position in `view_set`. `None` has the
  [reference-view rule](reference-view.md) pick one from the renders at the
  starting keypoints, the same resolution the localizer makes
  (`keypoint_localize::resolve_reference`), so the two kernels align a point's
  views to the same template.
- **`refine_view_against_reference` refines one view and nothing else.** Adding
  an observation to an existing track
  ([add-image-to-tracks.md](../reconstruction/add-image-to-tracks.md)) has a
  template already, either the point's stored bitmap or its reference
  observation, and only the new view's keypoint is in question.
  `ReferenceTemplate` names those two cases, so the caller does not render the
  reference itself. It returns `None` when the bitmap is not on the
  refinement's grid, the reference does not render in frame or has a different
  channel count from the target, the target does not project into its frame,
  or the target's core is out of frame at its seed.
- **`fuse_patch_bitmap` is the same kernel with nothing moving** (below), for a
  caller whose keypoints are settled and that wants only the fused mean.
- **The batch form is parallel over points** (rayon). Its `Progress` receives a
  `patches` count, is polled for cancellation before each patch, and, when
  detailed, times each sampler's renders in its own detail phase. The PyO3
  binding inlines its own parallel loop so it can build each point's seed slice
  lazily.

**Example.** Refine a point's views from the localizer's keypoints, with the
first view as the reference, and render its stored bitmap:

```rust
let params = KeypointSubpixelParams { render_bitmaps: true, ..Default::default() };
let refined = refine_patch_keypoints(
    patch, views, &localized.views,
    Some(&seeds),   // the localizer's keypoints, one Some per view
    Some(0),        // localized.views[0] is the reference observation
    &params,
);
assert_eq!(refined.reference, Some(0));
assert_eq!(refined.scores[0], 1.0);
```

## Contract and scope

This is a **local refiner**, and its guarantees hold only inside that scope:

- **Precondition: the seed is close.** Each input keypoint must already lie
  within the local convergence basin of the optimum (in practice ≲ 1 px). The
  algorithm linearizes around the seed; a seed outside the basin can converge
  to a wrong local optimum or be rejected, not rescued. Putting the keypoint in
  the basin is the caller's job (for example the localizer's discrete search).
- **Seeds come from the reconstruction.** `refine_keypoints` requires starting
  keypoints. On an `embedded_patches` reconstruction it seeds each view at that
  observation's stored keypoint, falling back per view to the reprojected
  centre for views carrying no observation (ones `select_views` admitted beyond
  the SIFT track). On a `sift_files` reconstruction, which stores none, a call
  without explicit `starting_keypoints` raises `ValueError`: a projection alone
  is not a keypoint for a local refiner, so convert to `embedded_patches` first
  or supply per-point seeds. (`refine_normals` is a normal optimizer rather
  than a local refiner, so its `use_stored_keypoints` flag stays a plain bool
  defaulting to `True`; explicit `False` anchors every view at the reprojected
  centre regardless of reconstruction kind, which is what `sfm compare --strips`
  and the cross-validation script want as a defined comparison reference.)
- **Scope: local only.** It searches no grid and visits no distant candidates;
  it takes a few Gauss–Newton steps from the seed.
- **Does not change membership, with one exception.** It never adds a view, and
  the only drop is the projection gate: a view in which the patch centre fails
  to project (behind the camera or outside the frame) is dropped, since the
  per-view offset is reported relative to that projection. A repeated image in
  the view set is refined once.
- **The reference is not moved.** Its keypoint is returned exactly as given,
  with score `1.0`. Whatever error it carries is shared by every view aligned to
  it, so it moves the triangulated point rather than adding reprojection error;
  see [patch-keypoint-localization.md](patch-keypoint-localization.md#the-template-the-reference-render).
- **Never worse than the seed.** A step is accepted only if it raises the ECC
  score against the template and stays in frame; if none does, the seed is
  kept. The template does not change while the views move, so the final score
  of every view is at least its score at the seed.

## Objective

Each view other than the reference maximizes its **Enhanced Correlation
Coefficient** against the template `T`, over its 2-DOF in-plane offset `δ_v`:

```
S_v(δ_v) = Σ_k w_k · ẑ(I_v(x_k(δ_v)))_k · T_k
```

where `k` runs over the `R×R` window support, `w_k` are the window weights,
`x_k(δ)` is the source sample point for grid pixel `k` at offset `δ`, and `ẑ`
is the weighted z-normalization. ECC is a zero-mean normalized correlation, so
it is invariant to a view's brightness and contrast (Psarakis & Evangelidis,
"An Enhanced Correlation-Based Method for Stereo Correspondence with Sub-Pixel
Accuracy," ICCV 2005; generalized to parametric alignment in Evangelidis &
Psarakis 2008). It is the continuous form of the score the localizer's grid
search maximizes.

**The template is the reference render**, fixed for the whole pass:

- the point's reference observation's core at its own starting keypoint,
  rendered with that observation's sampler; or
- where the point has no reference observation and the reference-view rule
  picks none it would store, the **fused mean** of the views at their starting
  keypoints (below), with no view held fixed.

It is never blurred: a blurred template places views no closer and lowers the
peak's curvature ([sharper-patch-bitmap.md](../../drafts/sharper-patch-bitmap.md)
Part 6). Since `T` does not depend on the offsets, the views are independent of
each other, and each is refined **once**. Where nothing renders to align to
(fewer than two views project, or the reference's core is out of frame), every
view keeps its seed, unscored.

## Algorithm: ECC (forward-additive Gauss–Newton)

For each view other than the reference, from its seed:

1. Sample the view's core at the current `δ` (fractional, through the
   view's sampler) and z-normalize it; form the ECC residual against `T`.
2. Take the ECC Gauss–Newton step on the 2-DOF offset. ECC's formulation
   handles the `ẑ` normalization analytically inside the step. The image
   Jacobian is `∂I/∂δ = ∇_src I · J`, where `∇_src I` comes from the sampler's
   analytic value-and-gradient interface ("Design details" below) and
   `J = ∂(image coords)/∂(patch grid)` is the warp Jacobian
   (`WarpMap::get_jacobian`), so the gradient is taken at the same level of
   detail as the value, with one render per step.
3. **Guard.** Accept the step only if it raises the score and keeps every
   sample in frame; otherwise backtrack by `line_search_shrink`, up to
   `line_search_max` times, and keep the current `δ` if no step improves. A
   step that would carry the view more than `max_offset_px` grid px from its
   seed is rejected. A near-singular system (low texture, the aperture
   problem) keeps the seed.
4. Stop when the accepted step is below `convergence_px` or after
   `max_gn_steps` steps.

Output per view: the refined `δ_v` mapped to `keypoint_v` by projecting the
re-anchored patch centre, as the localizer does
([patch-keypoint-localization.md](patch-keypoint-localization.md#mapping-a-shift-to-a-keypoint)).

## Parameters (defaults)

| parameter | default | meaning |
|---|---|---|
| `resolution` | 24 | the `R×R` grid the template and the per-view ECC are scored on |
| `window` | `GaussianDisk { sigma: 0.6 }` | per-pixel scoring weight and support |
| `sampler` | the sampler rule | which sampler renders each view, chosen once per view at its seed keypoint ("Sampling") |
| `robust_iters` | 3 | IRLS passes for the fused mean, where it is the template and the stored bitmap |
| `max_gn_steps` | 10 | Gauss–Newton steps per view |
| `convergence_px` | 0.01 | stop once an accepted step is below this, patch-grid px |
| `max_offset_px` | 2 | the furthest a view may move from its seed, patch-grid px |
| `line_search_shrink` | 0.5 | backtracking factor for a rejected step |
| `line_search_max` | 8 | backtracking attempts before a step is abandoned |
| `render_bitmaps` | `false` | also render the point's stored bitmap ("Outputs") |

## Sampling

The refiner samples each view's core from the **source pyramid** at the current
fractional `δ`, through the existing render machinery.

- **Interpolation** is the `SamplerChoice` every patch kernel takes: the sampler
  rule by default, or one `Sampler` (`Bilinear` / `BilinearMip` /
  `Anisotropic`) for every view. Anisotropic suits grazing or foreshortened
  views where a single tap under-samples; `BilinearMip` (one bilinear tap from
  the mip level nearest the warp's compression) bounds cross-scale aliasing at
  about bilinear cost; `Bilinear` is the cheapest tap. The rule picks
  `Anisotropic` for a view whose less compressed axis the single mip tap would
  read too coarsely and `BilinearMip` otherwise
  ([image-warping.md](../camera/image-warping.md) § "Choosing the sampler per
  view"). **Each view's sampler is chosen once**, for the observation at its
  seed keypoint, and the refine tile, every core read and the fused mean use
  it: the core only slides in the patch's plane, so its Jacobian, and the
  choice, hardly change as it moves. The reference's tile the stored bitmap is
  taken from (`render_view_tile`) applies the same rule on its own, at its
  keypoint.
- **Rendering** reuses `WarpMap::from_patch` and `remap_bilinear` /
  `remap_bilinear_mip` / `remap_aniso_with_pyramid`. Gradients come from the
  value-and-gradient variants of those functions ("Design details"), giving the
  analytic image gradient at the same interpolation and level of detail as the
  value, with no finite differences (which across a mip boundary would mix
  levels).

### Render-once context tile

Within one refinement the patch frame is fixed and only each view's 2-DOF
offset moves, so every render of a (point, view) pair is the same patch→image
map at a sub-pixel shift, and the solver evaluates about 10 to 14 of them per
pair (Gauss–Newton steps and line-search probes). The refiner therefore
prerenders **one** expanded, patch-grid-aligned tile per pair (`RefineTile`),
the localizer's context-tile idea
([keypoint-localization-search-cache.md](keypoint-localization-search-cache.md))
adapted to fractional reads:

- The tile is centred at the view's **seed** offset and sized
  `R + 2·(⌈max_offset_px⌉ + 2)`, so every offset the line search can accept
  reads in bounds (an out-of-coverage read falls back to a direct render, which
  in practice does not happen). It stores the sampler's unquantized values
  **and** the pre-composed patch-grid image Jacobian `∇_src I · J` per texel,
  the same analytic gradient the direct path computes, so a Gauss–Newton step
  is a tile read, not a render.
- Planes are stored as **cubic B-spline coefficients** (Unser's IIR prefilter),
  and reads evaluate the cardinal cubic spline (a 4×4 kernel; integer shifts
  reproduce texels exactly). The interpolator matters: bilinear reads'
  phase-dependent smoothing displaced the ECC optimum by up to about 0.065 px
  (pixel locking), and Catmull-Rom still left about 0.02–0.045 px on
  near-Nyquist content; the prefiltered spline keeps the planted-offset
  recovery inside the < 0.02 px target ("Validation"). A 2× supersampled
  bilinear tile was measured and rejected: it quadruples the prerender, which
  becomes the dominant cost.
- **Coarse-grid gate.** A tile is built only when the patch grid samples at
  least as densely as the source (≤ about 1.2 source px per grid px, estimated
  from the projected core corners). A coarser grid would freeze the sampling
  phase of source content above the grid's Nyquist limit, which direct
  rendering samples continuously (measured as spurious displacement on a
  coarse-grid synthetic fixture), and a tile supersampled to the source's
  density costs more than the direct renders it replaces, so coarse views
  (large-scale keypoints, about 28% of dino's pairs) keep the exact direct
  path.
- **Measured** (dino, 85 images, 46k points, two rounds of `sfm
  embed-patches`): sub-pixel CPU time 596 → 401 s and wall time 18.6 → 12.6 s
  in round 1; command wall time 101 → 89 s. Membership and point survival
  against the direct-render path are identical (0.047% churn over two rounds,
  none in a single round); keypoints differ by a median of 0.03 px, with about
  2% more than 0.5 px apart on weakly determined patches. Re-scoring both
  keypoint sets with the same scorer shows the tile run's final ECC equal or
  better (mean +0.00015 overall; on the tail 88.9% of the moved observations
  score higher, mean +0.0028): the smoother spline value and gradient let the
  solve converge further where the `u8`-quantized direct evaluations stalled.

## Outputs

`KeypointRefinement` holds parallel arrays over the views, in the input order:

- `views`: the image indices, deduplicated, after the projection gate;
- `keypoints`: the refined keypoint per view, source px; the reference's is its
  starting keypoint, exactly, and a view whose solve the guard refused stays at
  its seed;
- `offsets_px`: the keypoint's distance from the point's projection, source px;
- `scores`: the final ECC score against the template: `1.0` for the reference,
  `NaN` where the view could not be scored (fewer than two views, or no
  template rendered);
- `reference`: the reference observation the views were aligned to, as an index
  into `views`, or `None` where the template was the fused mean or there was
  none. It is set only when `render_bitmaps` is on, since it names the view
  `representative` is the tile of;
- `representative`: with `render_bitmaps` (the binding's
  `refine_keypoints(render_bitmaps=True)`), the point's **stored bitmap**,
  `R·R·4` RGBA, as [reference-view.md](reference-view.md#the-stored-bitmap)
  describes: the reference's `R×R` tile at its keypoint, which the refinement
  did not move, or, where there is no reference, the fused mean of the views at
  their final keypoints.

Two properties of the bitmap matter to consumers:

- **Each view is rendered with the refine `sampler`**, the sampler rule's
  choice by default, the same sampler its refinement read it with.
- **`None` is the uniform culled-point signal.** A point with fewer than two
  views after the projection gate has no bitmap, and so does a point with no
  reference where fewer than two views render in frame for the fused mean,
  finite and infinity alike (a `w = 0` point renders through the same path and
  gets a real bitmap, not a zero row). `sfm embed-patches` **drops** such points
  instead of keeping them with an all-black bitmap.

How `sfm embed-patches` records each point's reference observation and renders
its stored bitmap from it is in
[embed-patches-command.md](../../cli/reconstruction/embed-patches-command.md).

### The fused mean

Where the point has no reference observation and the reference-view rule picks
no view (no candidate has a self-similarity reading, or every view sees the
patch edge on or from behind), or reaches its pick only through its last
fallback, `without_any` (no view passed the coverage and clipping tests), the
stored bitmap is the views' fused mean: the views are rendered at their final
offsets, IRLS view weights are built from those cores, and the kept views are
rendered whole (`PatchViewStack`) and fused, weighted-mean RGB and
agreement·coverage alpha, the fusion normal refinement uses
(`PatchViewStack::fuse`, shared between the two modules). The mean weights each
view by its agreement with the mean alone, which favours views as blurry as the
mean over sharper ones, which is why it is the fallback rather than the stored
bitmap ([../../drafts/sharper-patch-bitmap.md](../../drafts/sharper-patch-bitmap.md)).
The same mean, at the starting keypoints, is the template where there is no
reference.

**Fusing without refining.** `fuse_patch_bitmap(patch, views, view_set,
keypoints, params)` renders the fused mean for a patch whose keypoints are
settled: the kernel above with `max_gn_steps = 0` and the mean asked for,
seeded at `keypoints`, so nothing moves, no reference is resolved, and the pass
only renders and blends. It returns `None` when fewer than two views render in
frame. The stored bitmap of settled keypoints is
[`render_patch_bitmap`](../../../crates/sfmtool-core/src/patch/stored_bitmap.rs),
which calls it as its fallback, and `render_patch_cloud_bitmaps` is its
whole-cloud form, parallel over points: `PatchCloud.render_bitmaps`, `sfm xform
--add-patch-bitmaps`, the viewer's display bitmaps for a file that carries
patch frames and no bitmaps, and a bench fit
([reference-view.md](reference-view.md#the-stored-bitmap)).

## Validation

It produces a better sub-pixel result than a parabolic or discrete estimate, so
it is not validated by equivalence to one:

- **Synthetic recovery.** On rendered views with a known planted sub-pixel
  offset (seed within the basin), recover `δ` to < 0.02 px when the sampler
  reads pyramid level 0: every `Sampler` at a patch grid at least as dense as
  the source, and `Bilinear` at any density.

  Above level 0 the target loosens with the level, because `δ` is located on a
  reconstruction whose samples are `2^ℓ` source px apart. A single bilinear tap
  reconstructs a piecewise-bilinear field whose amplitude response varies with
  the sub-sample phase; two views at different phases therefore correlate best
  slightly off the true offset, biased toward the level's own sample positions,
  by an amount that scales with the sample spacing. `BilinearMip` selects
  `ℓ = round(log₂ σ_major)` from the patch-grid → source scale, so a patch that
  minifies the source by ≳ 1.4× is refined on level ≥ 1 and recovers a planted
  offset a few percent short (a two-view fixture at `σ_major = 2.6` recovers a
  planted 0.78 px as 0.755 px; the equivalent `Bilinear` run, reading level 0,
  recovers 0.777 px). This is the accuracy side of the aliasing trade the level
  choice makes, not a defect of the mip composition: refining through level `ℓ`
  is numerically identical to refining a natively `2^ℓ`-downsampled image pair
  through `Bilinear`. `Anisotropic` blends two adjacent levels, which partly
  averages the per-level bias out.
- **Quality, not equivalence.** Refined keypoints are as good as or better than
  the seed by their score against the template and by reprojection.
- **The reference is unmoved.** Its keypoint comes back bit-identical, with
  score `1.0`.
- **Points at infinity.** A `w = 0` patch is refined, not skipped: synthetic
  recovery and the never-worse guard hold for it as for a finite patch. (The
  warp and the projection already handle `w = 0`, so the same objective,
  sampling and Jacobian apply; normal refinement, by contrast, skips such
  patches.)
- **Guard correctness.** A flat, out-of-frame or non-improving case keeps the
  seed. A seed outside the basin is a contract violation the refiner does not
  detect: it may converge to a wrong nearby optimum.

## Composition

- **The producer:** keypoint localization
  ([patch-keypoint-localization.md](patch-keypoint-localization.md), with its
  search in [keypoint-localization-search-cache.md](keypoint-localization-search-cache.md))
  aligns each view to the same reference render on the integer grid and lands
  each keypoint in the basin. The dependency is one way: this refiner needs
  only a patch, views, seeds and a reference.
- **Consumers:** `sfm embed-patches` runs the refiner after the localizer and
  writes the refined keypoints as the per-observation `keypoints_xy` (the
  geometric anchor is defined in
  [sfmr-file-format.md](../../formats/sfmr-file-format.md)); see
  [embed-patches-command.md](../../cli/reconstruction/embed-patches-command.md).
  Candidate spawning runs it on a spawned point when `SpawnParams::refine_subpixel`
  is set, against the localizer's reference. Add Image to Tracks refines its
  new view with `refine_view_against_reference`.

## Open questions

- **Inverse-compositional ECC** (precompute the template Hessian once instead
  of per step, the Lucas–Kanade variants of Baker & Matthews, IJCV 2004). The
  template is fixed for the whole pass, so the template-side precompute would
  be made once per point; whether it pays against the forward-additive step,
  whose per-step gradient is already cheap through the sampler's Jacobian, is
  not measured.

## Design details: the gradient-capable sampling interface

Per support pixel the Gauss–Newton step needs the value **and** `∂I/∂δ` (a 1×2
per channel) to accumulate the ECC normal equations. By the chain rule
`∂I/∂δ = ∇_src I · J`, where `∇_src I` is the image gradient in source-pixel
coordinates (from the interpolation) and `J = ∂(source coords)/∂(patch grid)`
is the warp Jacobian: `δ` is an in-plane patch-grid translation, so
`∂(source)/∂δ = J`. The pieces are in
[remap.rs](../../../crates/sfmtool-core/src/camera/remap.rs) and
[warp_map.rs](../../../crates/sfmtool-core/src/camera/warp_map.rs):

1. **Bilinear value and gradient**, `sample_bilinear_with_grad_u8(img, x, y, ch)
   -> (val, dI_dx, dI_dy)`, from the same four taps `v00, v10, v01, v11` as the
   value, with no extra fetch:

   ```
   dI_dx = (1-fy)·(v10 − v00) + fy·(v11 − v01)
   dI_dy = (1-fx)·(v01 − v00) + fx·(v11 − v10)
   ```

   `remap_bilinear_with_grad` and `remap_bilinear_mip_with_grad` apply it over
   a warp map.
2. **Anisotropic and pyramid value and gradient**, `remap_aniso_with_grad` (and
   the per-pixel `sample_aniso_with_grad`), computed at the **same level(s) and
   footprint as the value**: the per-level bilinear gradient is divided by the
   level's `2^level` to express it in level-0 source px (`x_level = x_0 /
   2^level`, so `∂I/∂x_0 = (∂I/∂x_level) / 2^level`), and the two levels are
   blended with the same `frac` the value uses. Finite differences across a mip
   boundary would mix levels, which is the reason to compute the gradient
   inside the sampler.
3. **The warp Jacobian**, `WarpMap::get_jacobian(col, row) -> [[f32; 2]; 2]`,
   the per-pixel `J` that `compute_svd` forms by central differences on the
   warp coordinates.

The refiner composes `∇_src I · J` per support pixel and channel; value-only
callers keep the plain `remap_*` functions.
