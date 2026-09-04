# Cluster-Patch Refinement

## Purpose

A feature matcher can decide that a handful of detections spread across a
handful of images probably came from the same point on a surface, but it says
almost nothing about how that piece of surface *looks* from each of those
images. Cluster-patch refinement fills that in, before any camera pose exists.
Given one such group of detections and the pictures they came from, it elects
one member as the group's reference, cuts a small square of image around it,
and then, for every other member, searches for the affine transform of that
square which best reproduces what the member's image actually shows. What comes
back, per member, is a photometrically verified local shape, a corrected
keypoint position, and a verdict on whether the member belongs at all.

Doing this early pays twice: a later stage can read a member's image-space
extent and refined position directly, and members that agree on a descriptor
while disagreeing on appearance become visible without a reconstruction.

This document specifies the kernel: `sfmtool-core`'s `patch::cluster_refine`,
its PyO3 binding, and the numerics. The kernel is pure — no I/O, no `.sift`
reads. The caller hands it decoded image pyramids, the SIFT geometry of every
image, and the clusters in CSR form; it hands back member-parallel arrays that
map 1:1 onto the `cluster_patches/` section. The motivation, the rejected
alternatives and the measured calibration behind the defaults live in
[cluster-patches.md](cluster-patches.md); the on-disk sections are specified
normatively in
[matches-file-format.md](../../formats/matches-file-format.md); the command is
[`sfm cluster-patches`](../../cli/image-feature/cluster-patches-command.md).

## Rust API

```rust
/// Per-member verdict. Discriminants match `matches_format::ClusterMemberStatus`.
#[repr(u8)]
pub enum MemberStatus {
    Reference = 0,
    Kept = 1,
    RejectedLowZncc = 2,
    RejectedShift = 3,
    DuplicateImage = 4,
    NotEvaluated = 5,
    RejectedUnlocalizable = 6,
}

/// `reference_members` entry for a cluster with no usable reference.
pub const REFERENCE_UNREFINABLE: u32 = u32::MAX;

pub struct ClusterRefineParams { /* … see Parameters */ }

/// One image's SIFT feature geometry (borrowed views of the `.sift` arrays).
pub struct FeatureGeometry<'a> {
    pub positions_xy: ArrayView2<'a, f32>,   // (N, 2), source px
    pub affine_shapes: ArrayView3<'a, f32>,  // (N, 2, 2), keypoint frame → px
}

pub struct ClusterRefineResult {
    pub reference_members: Vec<u32>,          // (C,)
    pub member_status: Vec<MemberStatus>,     // (M,)
    pub member_positions: Array2<f64>,        // (M, 2) = p
    pub member_affine_shapes: Array3<f64>,    // (M, 2, 2) = S
    pub member_zncc: Vec<f32>,                // (M,), NaN if not evaluated
    pub member_shift_px: Vec<f32>,            // (M,), NaN if not evaluated
}

pub fn refine_cluster_patches(
    pyramids: &[ImageU8Pyramid],
    features: &[FeatureGeometry<'_>],   // parallel to `pyramids`
    cluster_starts: &[u32],             // (C+1,) CSR
    member_images: &[u32],              // (M,)
    member_features: &[u32],            // (M,)
    params: &ClusterRefineParams,
    progress: Option<&AtomicUsize>,     // one tick per finished cluster
) -> ClusterRefineResult;

/// The reconstruction-free contamination signal computed from the refined
/// warps; see [cluster-warp-consistency.md](cluster-warp-consistency.md).
pub fn warp_consistency_residuals(
    cluster_starts: &[u32],
    member_images: &[u32],
    member_status: &[MemberStatus],
    reference_members: &[u32],
    member_affine_shapes: ArrayView3<'_, f64>,  // (M, 2, 2)
    n_images: usize,
) -> Vec<f32>;
```

**Why this shape.** Pyramids and borrowed feature views rather than file paths,
because decoding and `.sift` reading belong to the orchestration layer that
already holds both — the separation `ProjectedImage` gives the other patch
kernels. Three flat CSR arrays in and member-parallel vectors out, rather than
anything nested, because that is the shape the clusters have on disk, in the
matcher's in-memory cluster set and in numpy; nesting would be repacked at every
boundary. Nothing returns a `Result` — a member that cannot be evaluated is
*data* (`NotEvaluated`, `RejectedUnlocalizable`, …), and the failures that
remain are caller bugs (non-parallel inputs, malformed CSR), which assert.
`warp_consistency_residuals` stands apart from the result because it is a fit
over a whole refined cluster set, and a caller that only wants warps should not
pay for it.

```rust
use sfmtool_core::camera::remap::{ImageU8, ImageU8Pyramid};
use sfmtool_core::patch::cluster_refine::{
    refine_cluster_patches, ClusterRefineParams, FeatureGeometry, MemberStatus,
};

let pyramids: Vec<ImageU8Pyramid> =
    images.iter().map(|im: &ImageU8| ImageU8Pyramid::build(im, 5)).collect();
// One entry per image, borrowing that image's (N, 2) and (N, 2, 2) f32 `.sift`
// arrays.
let features: Vec<FeatureGeometry<'_>> = sift
    .iter()
    .map(|s| FeatureGeometry {
        positions_xy: s.positions.view(),
        affine_shapes: s.shapes.view(),
    })
    .collect();

let out = refine_cluster_patches(
    &pyramids,
    &features,
    &[0u32, 3, 5],           // two clusters: members 0..3 and 3..5
    &[0u32, 1, 2, 0, 3],     // member images
    &[17u32, 42, 8, 91, 5],  // member features
    &ClusterRefineParams::default(),
    None,
);

// Member 1's image-space extent and refined position, if it survived vetting.
if out.member_status[1] == MemberStatus::Kept {
    let s = out.member_affine_shapes.index_axis(ndarray::Axis(0), 1);
    let extent = (s[[0, 0]].hypot(s[[1, 0]]), s[[0, 1]].hypot(s[[1, 1]]));
    let p = out.member_positions.index_axis(ndarray::Axis(0), 1);
    let position = [p[0], p[1]];
    let _ = (extent, position);
}
```

## Theory

### The template

Every SIFT detection carries a position and a 2×2 affine shape `A` mapping the
detector's canonical *keypoint frame* onto image pixels. The reference's patch
is therefore a square in keypoint-frame coordinates: `resolution²` samples at
pixel-center offsets spanning `[−radius, radius]` per axis
(`step = 2·radius / resolution`, offset `u = k·step + 0.5·step − radius`),
carried into the image by `x = pos_ref + A_ref · u`. At the default `radius = 6`
the template spans 12 keypoint-frame units per axis — SIFT's own ~12×
descriptor window, so it vets a member against roughly the texture the detector
judged characteristic of the feature.

Only the *windowed* support is sampled. The scoring window is normal
refinement's shared `PatchWindow`, whose sigma is in normalized patch
coordinates where the grid spans `[−1, 1]²`; the default
`GaussianDisk { sigma: 0.5 }` is a Gaussian of half the patch half-width
confined to the inscribed disk, so in-plane rotation is exactly free and grazing
corners cannot leak in. Each channel is sampled independently, z-normalized over
that window, and kept only if its windowed norm clears `FLAT_NORM_SQ_EPS` — a
channel flat under the window carries no information. If **any** support sample
falls outside the image, or every channel is flat, the candidate reference is
unusable and the next candidate is tried.

### The objective

The score is the window-weighted zero-mean normalized cross correlation of the
z-normalized template against the member image sampled through the current warp,
per surviving channel, averaged over the template's channel count. Channel
identity is preserved — the reference's channel *c* is only ever correlated
against the member's channel *c*, and a member with fewer channels contributes
nothing for the missing ones. A member channel flat under the window scores `0`
for that channel rather than a garbage ratio.

Sampling is all-or-nothing: if any support sample leaves the frame the
evaluation returns the worst possible score (`+1.0` on the negated objective)
rather than a partial correlation, so the optimizer retreats instead of being
rewarded for walking off the image. Partial support would make scores from
different warps incomparable — exactly the ordering the optimizer relies on.

### The warp and the cascade

The unknown is an affine map from the reference's patch onto the member's image,
seeded from the detectors themselves (`M₀ = A_mem · A_ref⁻¹`, anchored at the
two detections):

```
W(x) = pos_mem + t + (I + D) · M₀ · (x − pos_ref)
```

`t` is a translation in source pixels, `D` a 2×2 correction to the seed's linear
part. The family stops at affine because the calibration measured the true
patch-to-patch warp at SIFT-patch scale as affine to well under a hundredth of a
pixel on every dataset, fisheye included: perspective terms only overfit.

The search is a three-stage Nelder-Mead cascade, each stage seeded from the
previous stage's optimum:

| stage | parameters | seed |
|---|---|---|
| shift | `t` (2) | `t = 0` |
| similarity | `t, σ, φ` (4), `D = e^σ R(φ) − I`, `σ` clamped to ±1.5 | the shift optimum |
| affine | `t, D` (6) | the similarity optimum, `σ`/`φ` expanded into `D` |

Each simplex is seeded at `θ₀ + scale_i·e_i` with 0.5 px for translations and
0.05 for shape entries, under standard coefficients (reflect 1, expand 2,
contract 0.5, shrink 0.5). Starting at translation matters: the detections'
*positions* are the noisiest part of the seed, and letting the shape float
before the patch is centred spends evaluations chasing a mis-registered
template. The shift and similarity stages exist only to seed the affine stage,
so they stop on a looser tolerance (`intermediate_convergence`) than the stage
whose answer is stored (`convergence`). Every stage additionally stops on a
stall — no improvement of the best value by more than `stall_tol` for
`stall_iters` consecutive iterations. That exit is for the affine stage, whose
reflect-heavy 6-dim crawl on a flat objective shrinks its value *spread* far
more slowly than it stops making *progress*; without it most members ran to the
iteration cap long after the score stopped moving. There is no multi-view
congealing pass — at raw-cluster sizes it measurably adds nothing over pairwise
refinement.

### Pyramid levels

For every sampled image — the template's and each evaluation's — the level is
`ℓ = clamp(⌊log₂ s_min⌋, 0, L−1)`, where `s_min` is the smaller singular value
of the support map's linear part (the sample spacing in source pixels along the
compressed axis); the map is divided by `2^ℓ` before sampling. The level is
chosen **per objective evaluation**, not once per member, because `D` changes
the linear part as the cascade runs. Sampling from too fine a level aliases the
score surface the optimizer descends, a worse failure than the blur of a
slightly coarse one. Full anisotropic footprints would be more correct still and
are deliberately not used: this is a single tap from the selected level, the same
choice the other patch kernels make, differing only in the level rule
(`floor(log₂ s_min)` here against `round(log₂ σ_major)` there).

### Which member anchors, and which members are eligible

Before anything is refined, each member's own patch is scored for
*localizability* — the noise-normalized weak-axis positional uncertainty of its
ZNCC self-similarity surface
([patch-localizability.md](patch-localizability.md)) — on its full `resolution²`
grid at its own SIFT geometry, with the shared refinement window and the global
`σ_noise = 3.0`. A member whose `σ_pos` exceeds `max_keypoint_uncertainty`
becomes `RejectedUnlocalizable` and takes no further part: a flat wash or a
straight edge can neither anchor a cluster nor honestly join one, since it
agrees photometrically with any translation along its weak axis. The gate
samples with a nearest-valid-pixel clamp rather than an in-frame requirement, so
a member near the border is scored on its visible content instead of escaping
the gate; only non-finite geometry skips it, a `NaN` score keeps the member (the
`embed-patches` convention), and a threshold of `0` disables the gate.

The reference is then the surviving member with the largest SIFT scale
`√|det A|`, ties to the lowest global member index — a larger patch resolves the
smaller ones rather than the reverse. Selection is policy, not format: the
reference is stored as data, so a better policy can replace this one without a
format change. If the best candidate's template proves unusable the next
candidate by scale is tried, and a cluster where every candidate fails is
*unrefinable*: `reference_members[c] = REFERENCE_UNREFINABLE`, and every member
not already gated is `NotEvaluated`. The same follows when fewer than two
members survive validation, which also drops members with an out-of-range
feature index or a degenerate shape (`|det A| < 1e-9`).

### Acceptance

A refined member is vetted in a fixed order: ZNCC below `min_zncc` →
`RejectedLowZncc`; otherwise `|t|` above `max_shift_px` → `RejectedShift`;
otherwise kept. The ZNCC gate is deliberately permissive, because the measured
failure mode is over-culling rather than contamination and because the achieved
ZNCC routinely exceeds the *ground-truth* warp's own — the score gates match
validity, never warp correctness. Consumers re-gate on the stored signals, which
is why rejected members keep their measured ZNCC and shift.

Finally, at most one member per image survives: among provisionally kept members
sharing an image the highest ZNCC wins, ties to the lowest member index, and the
rest become `DuplicateImage` — as does any member sharing the reference's own
image, which is marked before it is ever refined.

### What is returned, and why it is absolute

The working unknown is the relative warp `W = (I + D)·M₀`, but the result
reports the **absolute affine shape** `S = W · A_ref` and the member's
**refined absolute keypoint position** `p = pos_mem + t`. `S` is literally the
matrix the winning evaluation sampled with, so the reported shape is the shape
that was measured, and because it maps the detector's canonical unit frame onto
that member's image pixels, its column norms are the member's image-space
extent: a consumer reads extent and position per member with no `.sift` file
open. The reference member's own entries are `A_ref` and its detected position,
so the relative warp stays recoverable as `W = S · S_ref⁻¹`, after which
`x_mem = W·(x − x_ref) + p`. That inversion is what keeps a *derived* file — one
whose reference member has been filtered out — meaningful: absolute shapes and
positions stay valid regardless, and only the relative reading needs the
reference.

The two arrays are member-parallel but not member-complete: a member the
cascade never fitted (`NotEvaluated`, `RejectedUnlocalizable`, and a
`DuplicateImage` that shared the reference's image) has an all-zero entry, and
`member_status` is what says so. The `.matches` writer reads that: it stores
the refinement's values for the measured members and leaves the input's
detections in place for the rest, so the file it writes has no holes (see
[`matches-file-format.md`](../../formats/matches-file-format.md), Member
geometry). Nothing about the geometry is stored in `cluster_patches/`, which
carries the vetting evidence alone.

### After refinement

`warp_consistency_residuals` factors all of a cluster's recovered warps against
one weak-perspective camera per image and one tangent frame per cluster, and
reports each member's misfit — a reconstruction-free contamination signal that
catches the wrong-match member which aligns photometrically on repetitive
texture ([cluster-warp-consistency.md](cluster-warp-consistency.md)). It is
computed in the same binding call and stored beside the ZNCC and shift, as a
signal and not a gate. Once tracks and poses exist, the sharper pairwise
agreement test of
[member-coherence-validation.md](member-coherence-validation.md) decides
membership on the same kind of evidence with a reconstruction in hand.

## Implementation notes

**Determinism is a contract, not an accident.** Clusters refine in parallel with
rayon over disjoint member ranges, per-cluster scratch lives inside the closure,
and every tie is broken by index: reference selection sorts by scale then global
member index, the per-image dedupe compares with a strict `>` so the earlier
member wins, the simplex reorder is a stable insertion sort, and the returned
optimum is the first minimum (the numpy-argmin convention). Two runs over the
same input are bit-identical under any thread schedule.

**The status discriminants are a cross-crate invariant.** `sfmtool-core` does not
depend on `matches-format`; the binding casts `MemberStatus` to `u8` and writes
it straight into the `cluster_patches/` section. The two enums must stay
numerically identical, and a new status has to land in both — plus the format's
validator — in one change.

**The tile bound is the frame test.** Each evaluation samples through a
per-(member, level) tile: a planar f32 copy of the touched region of that pyramid
level, built once when the level is first reached and grown lazily to cover each
evaluation's bounding box. Because the tile is always a subset of the image and
always covers the evaluation's *clipped* footprint, an out-of-tile tap is exactly
an out-of-frame tap — the lane-bounds test doubles as the all-in-frame test and
as the early-out, and the hot loop needs no separate image-bounds check.

**The tile is centered for a numerical reason.** Windowed ZNCC needs a variance,
accumulated as `S2 − S1²/W` in f32. On raw 0–255 intensities that difference
cancels catastrophically for low-contrast patches; subtracting the tile mean at
conversion time keeps the accumulation well-scaled, and the windowed ZNCC is
shift-invariant so nothing has to be undone afterwards.

**Scalar and AVX2 must agree, and the agreement is structural.** The scalar path
is the reference implementation, the non-x86 fallback and the dual-path test's
oracle; the runtime-dispatched AVX2+FMA path is a restructuring, not a
re-derivation. It fuses all template channels into one k-major pass so the
channel-invariant work — coordinate FMAs, the in-frame mask, tap indices, blend
weights — is computed once per 8 support points, while keeping each channel's
accumulation order identical to the channel-major original, which is what makes
the fusion bit-exact rather than merely close. The four bilinear taps come from
64-bit pair loads, since the two horizontal taps of a lane are adjacent floats:
half the fetched elements of a 32-bit hardware gather, and plain loads sidestep
the microcoded gather penalty on hybrid parts, where most rayon threads land.
Non-finite coordinates from a degenerate warp convert to `i32::MIN` and are
caught by the same lane mask; the scalar path checks them explicitly.

**The sampler is local on purpose.** The kernel reuses the house pixel-center
bilinear convention, the shared window support and the `weighted_moments_pub` /
`znorm_write` z-normalization — but not `view_selection`'s
`sample_support_affine`, whose contract (border-gated maps, no validity
reporting, `u8` re-rounding for remap parity) is incompatible with a sampler that
must report out-of-frame and keep continuous values. Nor is
`score_raw_against_reference` shared: the fused sample+reduce loop realizes the
same algebra in one pass, and splitting it would reintroduce the intermediate
buffer the fusion exists to remove.

**The optimizer allocates nothing.** The simplex lives in fixed `[f64; 6]`
buffers (the affine stage is the widest) and the per-iteration reorder is an
in-place stable insertion sort. At the order of 10⁸ objective evaluations in a
full run, the per-iteration `Vec` churn of the original transcription cost about
as much as the arithmetic; removing it changed no result.

**Profiling is opt-in and free when off.** `cluster_refine::prof` carries the
house phase timers and counters — gate, template, cascade, tile builds and their
pixel volume, evaluations per cascade stage — behind `SFMTOOL_PROFILE=1`,
compiling to one branch on a cached flag when the variable is unset.

### Measured cost

On dino_dog_toy (85 images at 2040×1536, 105,326 clusters, 373,194 members,
i9-14900HX, 32 threads) the kernel runs in 3.2 s wall / 103 CPU-s, and the whole
`sfm cluster-patches` invocation in 7.8 s. Three changes took it there from
6.1 s / 194 CPU-s: the fused pair-load AVX2 kernel (1.93 → 1.15 µs per objective
evaluation, bit-identical), the allocation-free Nelder-Mead (−30 CPU-s,
bit-identical), and the cascade stopping rules — the only one of the three that
changes results. Their sweep: evaluations 265 → 213 per member, kept members
+0.03%, mean kept ZNCC −0.0001, warp-consistency median 0.0677 → 0.0669 and p90
0.1993 → 0.1972 (slightly better), status flips 0.76%. A tighter
`stall_tol = 2e-4` was rejected because it broke the synthetic scale-1.25 /
rotation-20° recovery case.

Roughly 60% of kernel CPU is still the objective, most of it the affine stage's
cap-bound crawl. Two candidates were considered and not pursued: replacing that
stage with a Gauss-Newton/ECC step on an analytic windowed-ZNCC gradient, and
luminance-only refinement (3× fewer channel passes, but it changes matching
semantics and needs its own quality study).

## Parameters

Defaults are `ClusterRefineParams::default()` in
`crates/sfmtool-core/src/patch/cluster_refine/params.rs`, except for the three
module constants the last column marks. The CLI's `--patch-size` is the **full**
template edge length while the kernel's `radius` is a half-width;
`src/sfmtool/_cluster_patches.py` is the sole conversion site
(`radius = patch_size / 2`).

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `radius` | `6.0` | Template half-width, keypoint-frame units (CLI `--patch-size 12.0`) |
| `resolution` | `25` | Support samples per axis; the kernel clamps up to 2, the CLI to ≥ 3 |
| `window` | `GaussianDisk { sigma: 0.5 }` | Scoring window; sigma in normalized patch coordinates, where the grid spans `[−1, 1]²` |
| `min_zncc` | `0.85` | Acceptance threshold on the achieved windowed ZNCC |
| `max_shift_px` | `3.0` | Max translation drift from the SIFT seed, source px |
| `max_keypoint_uncertainty` | `0.35` | Localizability gate on `σ_pos`, template-grid px; `0` disables it |
| `max_iters` | `120` | Nelder-Mead iterations per cascade stage |
| `convergence` | `1e-5` | Simplex value-spread stop, affine stage |
| `intermediate_convergence` | `1e-4` | …and for the shift and similarity stages, which only seed the next |
| `stall_iters` | `20` | Iterations without progress before a stage exits |
| `stall_tol` | `1e-4` | Best-value improvement (ZNCC units) that counts as progress |
| `MIN_ABS_DET` | `1e-9` | Floor on a usable SIFT shape's `det A` magnitude (`mod.rs`) |
| `SIGMA_CLAMP` | `1.5` | Log-scale clamp of the similarity stage (`mod.rs`) |
| `LOCALIZABILITY_SIGMA_NOISE` | `3.0` | Photometric-noise constant of the gate, intensity units (`mod.rs`) |

## Python bindings

`_sfmtool.matching.refine_cluster_patches`, in
`crates/sfmtool-py/src/matching/cluster.rs`, registered beside
`background_floor_clusters` and `clusters_to_pair_matches`:

```python
refine_cluster_patches(
    images, positions, affine_shapes,
    cluster_starts, member_images, member_features, *,
    radius=6.0, resolution=25,
    window="gaussian_disk", window_sigma=None,
    min_zncc=0.85, max_shift_px=3.0,
    max_keypoint_uncertainty=0.35,
    max_iters=120, progress=None,
) -> dict
```

`images` is one `HxW` or `HxWxC` uint8 array per image, in the images-section
order the cluster arrays index; `positions` and `affine_shapes` are parallel
lists of `(N, 2)` and `(N, 2, 2)` float32 arrays; the three cluster arrays are
uint32. `window` is `"gaussian_disk"` (default), `"gaussian"` or `"uniform"`, and
`window_sigma=None` resolves to `0.5`. The four cascade-tuning knobs
(`convergence`, `intermediate_convergence`, `stall_iters`, `stall_tol`) are
deliberately not exposed: they trade evaluations against a kept set that has been
swept once, so moving them changes results rather than only cost.

Argument names match the Rust ones. Parallel-list lengths, per-image array
shapes, CSR self-consistency and the `member_images` range raise `ValueError`
before the GIL is released; out-of-range `member_features` and degenerate shapes
are data, not errors, and reach the caller as `not_evaluated`. Pyramids are built
through `patches::views::build_pyramids_from_image_list`, and the kernel runs
under `py.detach`.

The returned dict is member-parallel: `reference_members` `(C,)` uint32
(`0xFFFFFFFF` = unrefinable), `member_status` `(M,)` uint8,
`member_positions` `(M, 2)` float64, `member_affine_shapes` `(M, 2, 2)`
float64, `member_zncc` `(M,)` float32, `member_shift_px` `(M,)` float32, and
`member_consistency_residual` `(M,)` float32 — the warp-consistency signal,
computed inside the same call.

```python
from sfmtool._sfmtool.matching import refine_cluster_patches

out = refine_cluster_patches(
    images, positions, affine_shapes,
    cluster_starts, member_images, member_features,
    radius=6.0, min_zncc=0.85,
)
kept = out["member_status"] == 1                    # 1 == kept
shapes = out["member_affine_shapes"][kept]          # absolute 2x2 shapes
points = out["member_positions"][kept]              # refined positions
```

## Testing

`cluster_refine/tests.rs` covers the kernel: **synthetic recovery** across the
calibrated warp range (scale 0.8–1.5×, rotation ≤ 20°, shear ≤ 0.15) with the
seed perturbed by the experiment-observed noise (`|Δlog s|` 0.07, `|Δrot|` 4°,
1 px shift), recovering `W = S·S_ref⁻¹` through the reference member exactly as a
consumer would and asserting a support-grid RMSE around 0.3 px; **one test per
gate** (flat member image → `RejectedLowZncc`; seed drifted past `max_shift_px`
→ `RejectedShift`; support out of frame → `NotEvaluated`; an unlocalizable
member excluded, and unable to become the reference; a border member still
scored through the clamped sampling; every reference candidate out of frame →
unrefinable; a degenerate cluster → not evaluated; two members in one image →
exactly one `Kept`); **determinism**, two runs bit-identical; and a **dual-path**
check that AVX2 and scalar scores agree within 1e-4. That the low-ZNCC test uses
a *flat* member image rather than an unrelated smooth texture is behaviour, not
test convenience: over the ~50 effective samples of the window the affine
optimizer can chase an unrelated smooth texture to a spurious ZNCC above the
permissive 0.85 gate, which then trips the shift gate instead.

`consistency/tests.rs` covers the residual fit (oracle cameras fit exactly,
absolute shapes reproduce the relative-warp residuals, a contaminated member
scores highest, non-participants NaN, runs deterministic).
`tests/rust_bindings/test_cluster_patches_rust_bindings.py` pins the dict schema,
dtypes, progress ticks and every `ValueError` path.
`tests/patch/test_cluster_patches.py` drives the command over the
`isolated_seoul_bull_17_images` fixture through the real pipeline
(`ws init` → `sift --extract` → `match --cluster` → `cluster-patches`) and
asserts that the output verifies, that over half the multi-member clusters keep
at least one member, and that statuses stay inside the enum.

## Non-goals

- **No matching.** The kernel refines the clusters it is given; producing them is
  [track-cluster matching](../features/track-cluster-matching.md)'s job, and a
  refinement-knob change must never force a re-match — the reason the two are
  staged artifacts.
- **No perspective warp, and no multi-view congealing pass.** Both were measured
  against the calibration data, and both are dead ends at this scale.
- **No reconstruction.** The operation runs before any pose exists, so the
  geometric consistency it can offer is the reconstruction-free residual, not a
  reprojection test.
- **No gate on consistency.** The residual is stored, never thresholded here.

## Open questions

- **The localizability threshold's unit is not resolution-invariant.**
  `max_keypoint_uncertainty` is in template-grid px, so the gate weakens as the
  template is sampled more finely: on dino_dog_toy, moving from 15 to 31 samples
  per axis at a fixed `0.35` cut `RejectedUnlocalizable` from 1,913 members to
  372. Since `--resolution` is freely tunable, one knob silently moves another
  gate's strength. Re-expressing the threshold in a resolution-independent unit
  (keypoint-frame or source px) would fix it, and would change the meaning of the
  current default.
- **Reference-selection policy.** Largest SIFT scale is the shipped policy and a
  known weakness on rig captures, where the largest-scale member is often an
  untracked feature. The format is policy-agnostic; the alternatives (template
  self-agreement, descriptor centrality) are the design spec's question to
  settle.
