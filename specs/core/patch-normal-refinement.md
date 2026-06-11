# Photometric Patch-Normal Refinement

**Status:** Proposed (core); a prototype lives in `scripts/patch_crossval.py`
(`--refine-normal`). Builds on `specs/core/patch-cloud.md`
(`OrientedPatch`, `WarpMap::from_patch`, `remap_*`).

## Problem

A reconstructed 3D point `X` is seen by cameras `{(Kᵢ, Tᵢ)}`. Its surface around
`X` is (locally) a small plane — the surfel. `PatchCloud::from_reconstruction`
gives that surfel an *initial* normal (the mean viewing direction), but that is
the camera-facing plane, not the true surface plane. We want the normal `n` that
maximizes **photometric consistency** across the views: the plane through `X`
whose rendered patches agree the most.

This is the planar-patch case of multi-view stereo (PMVS-style patch
optimization / plane-sweep): for a pinhole camera the patch→image map is a
homography `Hᵢ(n)` induced by the plane, and the inter-view map `Hⱼ(n)·Hᵢ(n)⁻¹`
aligns the images exactly when `n` is the true tangent plane. We use the general
per-pixel projection (`WarpMap::from_patch`), so distortion / fisheye work
unchanged.

## What is and isn't a degree of freedom

For a patch centered at fixed `X` with fixed world extent, rendered into the
**shared** canonical `(s, t)` grid:

- **Normal `n` (2 DOF):** the only parameter that changes which 3D plane is
  sampled, hence the only parameter that affects cross-view consistency. This is
  what we optimize.
- **In-plane rotation (1 DOF): a gauge.** Rotating `u`/`v` about `n`
  reparameterizes `(s, t) → world` *identically in every view*, so (with a
  radially-symmetric window) it leaves every pairwise NCC unchanged. We do **not**
  optimize it.
- **Center `X` / depth, and extent (scale):** fixed here (from triangulation and
  the keypoint scale). Optional future DOF — PMVS jointly refines `(depth,
  normal)`; depth refinement would move the point and is out of scope for a
  "refine the normal of a given point" routine.

So refinement is a **2-DOF search on the sphere** around the initial normal.

## Objective

Photoconsistency `Φ(n)` over the patches `{pᵢ(n)}` rendered into each view:

- **Aggregation.** Prototype uses the **mean pairwise NCC** over all `C(V,2)`
  view pairs. Alternatives: a **reference-view** mean (NCC of every view against
  one chosen reference — `O(V)` instead of `O(V²)`, PMVS-style) and **robust**
  aggregation (median / trimmed mean) to tolerate a few bad views.
- **Per-pair score.** **Per-channel ZNCC on color**, averaged over channels
  (Gaussian center weight). Normalizing each channel independently keeps the
  invariance to a *per-channel* affine (gain/offset), so it is robust to
  per-camera white-balance and exposure differences while still using
  chrominance — which discriminates surfaces that match in luminance but differ
  in colour (the prototype originally collapsed to luminance and lost this).
  Stacking the channels into one vector instead would assume a single shared
  affine across channels and is *less* white-balance robust. Census / MI are
  options for non-affine changes.
- **Validity.** A candidate normal can project the patch partly out of frame
  (NaN) or behind a camera in some views. Score only over commonly-valid pixels,
  and require a minimum valid fraction per view and a minimum number of valid
  views, else treat the view/candidate as invalid.

## Current prototype algorithm

```
refine_normal(center, init_n, up, half_extent, views, images):
    u, v = tangent_basis(init_n)
    grid = linspace(-tan(range), tan(range), steps)        # default range=25°, steps=7
    best = (init_n, Φ(init_n))
    for a in grid, b in grid:                              # steps² candidates
        n = normalize(init_n + a·u + b·v)
        if Φ(n) > best.score: best = (n, Φ(n))
    return best
Φ(n):  render patch under n into each view (from_patch + remap_bilinear at the
       validated resolution); return mean pairwise windowed-NCC.
```

Single-level dense grid; `steps²·V` renders per point. On seoul it lifts mean
pairwise NCC by ~0.03 on average and up to +0.17 on the least-consistent tracks.

## Proposed core API

```rust
pub struct NormalRefineParams {
    pub angular_range_deg: f64,   // half-extent of the search cone
    pub init_steps: u32,          // coarse grid resolution per axis
    pub refine_levels: u32,       // coarse-to-fine passes (each shrinks the cone)
    pub objective: Objective,     // AllPairsMean | ReferenceMean | RobustMedian
    pub min_valid_fraction: f64,  // per-view valid-pixel floor
    pub min_views: u32,
    pub anisotropic: bool,        // remap_aniso vs remap_bilinear
}

pub struct NormalRefineResult {
    pub normal: Vector3<f64>,
    pub photoconsistency: f64,
    pub init_photoconsistency: f64,
    pub valid_view_count: u32,
    pub confidence: f64,          // peakedness of Φ at the optimum (see below)
}

/// Refine one patch's normal. `views[i]` carries the camera, its world-to-camera
/// pose, and a source-image pyramid (built once, reused across candidates).
pub fn refine_patch_normal(
    center: Point3<f64>,
    init_normal: Vector3<f64>,
    up_hint: Vector3<f64>,
    half_extent: [f64; 2],
    views: &[PatchView<'_>],
    resolution: u32,
    params: &NormalRefineParams,
) -> NormalRefineResult;

/// Batch over a PatchCloud (parallel across patches).
pub fn refine_patch_cloud(cloud: &mut PatchCloud, views: ..., params: ...) -> Vec<NormalRefineResult>;
```

`refine_patch_normal` composes `WarpMap::from_patch` + `remap`; the per-source
`ImageU8Pyramid` is built once per view and reused for every candidate.

## Improvements to discuss

Ordered roughly by value.

1. **Coarse-to-fine search.** Replace the single dense grid with a few levels:
   a coarse grid over the full cone, then recenter on the best and shrink the
   cone, repeat. Same precision for far fewer evaluations (e.g. 3 levels × 5²
   ≈ 75 vs one 15² = 225), and less prone to skipping a narrow peak.

2. **Anti-aliased sampling (`remap_aniso`).** Oblique views foreshorten the
   patch; bilinear sampling then aliases and *biases the NCC downward*, which can
   push the optimum away from the true normal. Using `remap_aniso` (the patch
   warp's Jacobian SVD picks the pyramid level) de-aliases grazing views. Cost:
   build one pyramid per source image (already supported, `ImagePyramid`).

3. **Visibility / robustness.** Independent NCC over all views assumes every view
   sees the same surface. Improve by (a) dropping **back-facing** views
   (`is_front_facing`) and views past a grazing-angle cutoff; (b) **robust
   aggregation** (median / trimmed mean); (c) an **iterative good-view set** —
   refine, drop views whose NCC to the consensus is low (occluded / wrong
   surface), re-refine. This is the most impactful change for real scenes with
   occlusion.

4. **Gradient-based local step.** The objective is smooth in `n`; after the
   coarse grid, a Gauss-Newton / inverse-compositional (Lucas-Kanade) step
   converges in 1–3 iterations. The warp Jacobian `∂(source pixel)/∂n` factors as
   `∂project/∂world · ∂world/∂n`, and steepest-descent images per view can be
   precomputed (inverse-compositional). Much faster and more accurate than a grid
   near the optimum; more code. Likely pair: coarse grid (global-ish) → GN
   polish.

5. **Reference-view objective.** All-pairs mean is `O(V²)` renders-comparisons;
   a reference-view mean is `O(V)`. Choose the reference as the most
   fronto-parallel / highest-resolution view. Cheaper and usually as accurate;
   slightly less robust if the reference is itself bad (mitigated by good-view
   selection).

6. **Multiple initializations.** Repetitive texture makes `Φ` multi-modal. Seed
   the search from both the mean-viewing and the geometric (PCA) normals (and
   keep the better optimum) to avoid a wrong local max.

7. **Confidence / conditioning.** When all cameras are on one side (narrow
   baseline) the normal is weakly constrained — `Φ` is flat along the viewing
   direction. Report a **confidence** from the local curvature of `Φ` at the
   optimum (Hessian eigenvalues, or how much `Φ` drops over the search cone);
   downstream consumers can trust / filter normals by it. Degenerate (single-
   sided, ≤2 views) points should keep the initial normal and flag low
   confidence.

8. **Cloud-level smoothness (later).** Refining points independently can give
   noisy normals on weak points. A light prior (blend toward the mean normal of
   k-NN points) or a post-pass smoothing trades a little photoconsistency for
   spatial coherence. Out of scope for v1.

## Recommended v1

Coarse-to-fine grid (3 levels) + `remap_aniso` + back-face/grazing filtering +
robust (median) aggregation + a confidence score; reference-view objective for
speed; gradient polish and good-view iteration as fast-follows.

## Open questions

- **Objective:** all-pairs vs reference-view as the default? (Recommendation:
  reference-view with good-view selection.)
- **Stopping / cone schedule** for coarse-to-fine (levels, shrink factor, steps).
- **Confidence definition** and the threshold below which we keep the init normal.
- **Where extent/depth refinement** belongs — this routine, or a separate joint
  `(depth, normal)` optimizer.
- **Batch placement:** Rust `refine_patch_cloud` (parallel, pyramids per view) vs
  leaving orchestration to callers.
