# Patch-View Selection

Patch-view selection decides, for a single 3D point, which images actually see
the small piece of surface that point sits on. Given the point and its oriented
patch it returns the views that *photometrically* see that patch — the ones whose
pixels agree with the rest on what the patch looks like, not merely the ones the
geometry says it projects into. It is a standalone, separately-callable
algorithm, used by the [sift-based → patch-based reconstruction
pipeline](sift-to-patch-reconstruction.md) alongside
[normal refinement](patch-normal-refinement.md) and
[keypoint localization](patch-keypoint-localization.md).

## Problem

A reconstruction's track for a point records the views where its feature matched
— often not every view that actually sees the surface the point sits on.
Expanding the track to those additional views gives the patch more support to
register and score against. This algorithm picks those views by geometric
visibility (the point projects into the frame, the patch faces the camera) plus a
**photometric** check that their pixels actually agree on the patch, so
self-occluded or disagreeing views are left out.

## Inputs

- One 3D point `X_p` with its **patch frame** (`u_p`, `v_p`, normal `n_p`).
- The point's **track** — the views that already observe it — used to build the
  reference appearance, and always admitted.
- The reconstruction's camera poses + intrinsics, and the source images.

## Algorithm

1. **Candidates.** The track views plus every other image that *geometrically*
   sees the surfel — the patch is front-facing (`OrientedPatch::is_front_facing`),
   the point is in front of the camera, and the patch's projected footprint
   overlaps the frame (`view_could_see_patch`, § "The geometric cull") — and
   whose render covers the reference support.
2. **Reference appearance.** Render the patch in each track view and combine them
   into a robust consensus — a reference image of what the surface looks like,
   from the views that already observe the point.
3. **Admit.** Render each candidate's patch (under the point's normal) and
   correlate it (windowed ZNCC) against the reference; admit those that clear a
   threshold tied to the track's own self-agreement (`min_relative_zncc`). Track views
   are always admitted.

This rejects self-occluded / disagreeing views that pure geometric visibility
would wrongly include.

### Track-view anchoring (opt-in)

Steps 2 and 3's *track-view* half can be **anchored at the track views' stored
keypoints**: each track view's render is recentred in-plane so it samples where
that image's feature actually is, rather than where the point currently
reprojects. The reference is then fused from what was matched, and each track
view's own returned score is taken through the same anchored render, so a track
view's reprojection residual neither smears the reference nor deflates the score a
caller might evict it on. See
[member-coherence validation](member-coherence-validation.md#members-are-sampled-at-their-keypoints-not-at-the-reprojection)
for why charging a geometric residual to a photometric measure is the wrong
reading.

**Candidates are always scored at their projections.** A candidate is by
definition a view with no observation, so it has no keypoint; its score is
therefore "candidate at its projection against a reference fused at keypoints".
Track views with no stored keypoint (a `sift_files` reconstruction, an overridden
base list naming an image the point does not observe) fall back to projection
anchoring individually.

Anchoring is **off by default** at both the kernel and the binding
(`keypoint_anchor=False`), because it changes the reference every score is taken
against: callers that were selecting views keep their behaviour, and a caller that
wants the anchored reading asks for it. Keypoints handed in *at* the reprojections
reproduce the unanchored selection exactly, so the parameter is a strict
generalization rather than a second render path.

## Output

The selected **view set** `G` — the admitted views for the point (track views
first, then the photometrically-vetted candidates in ascending index order),
their per-view ZNCC to the reference, the track's self-agreement, and the
**track-view count**: how many leading admitted entries are track views, so a
consumer can split `G` by provenance without re-deriving the track. The
localizer's [consensus-basis pick](keypoint-localization-consensus-basis.md)
consumes both the scores and that count.

## Parameters (defaults)

| parameter | default | meaning |
|---|---|---|
| `min_relative_zncc` | 0.7 | admit a candidate whose windowed ZNCC to the reference clears this fraction of the track's own self-agreement |
| `min_self_agreement` | 0.3 | trust gate: when the track's self-agreement (its views' mean ZNCC to the reference) is below this, there is no trustworthy reference, so the track is admitted verbatim with **no** candidate expansion. At or above it, the admission bar is `min_relative_zncc × self_agreement` |
| `min_track_views` | 2 | minimum number of *valid* track views (those passing the per-view validity gate over the common support) needed to build a reference; a track below this admits its views verbatim with no vetting |

When the track's self-agreement is below `min_self_agreement` the track is
admitted **verbatim** (no candidates added). The bar for actual vetting is
therefore simply `min_relative_zncc × self_agreement`, evaluated only when
`self_agreement ≥ min_self_agreement` — the floor decides *whether* to expand,
not how the bar is computed.

## Implementation

The selector is `select_patch_views` / `select_patch_cloud_views` in
[view_selection.rs](../../../crates/sfmtool-core/src/patch/view_selection.rs),
bound as `PatchCloud.select_views` and called per point by the pipeline in
[_embed_patches.py](../../../src/sfmtool/_embed_patches.py). It reuses the patch
render + `is_front_facing` for candidacy and the IRLS consensus + windowed ZNCC
for the reference and scoring — the same machinery as normal refinement and
keypoint localization.

`select_patch_views` takes a `Progress` and answers
`Result<ViewSelection, Cancelled>`. It reports `build reference` and `score
views`, counts the views scored, and checks cancellation before and after
reference construction and between views — where a selection over a capture's
worth of images spends its time. A cancelled call returns `Cancelled` rather
than a partial view set, so an interactive caller never installs half a
selection.

There is one entry point rather than a reporting one beside a plain one,
because two would be two places for the gates to drift apart. A caller with no
one watching passes `Progress::none()`, which reports nothing and never
cancels. `select_patch_cloud_views` takes a `Progress` of its own, beside the
`done` counter a Python poller reads: it counts `patches`, polls for
cancellation before each patch (a cancelled batch returns `Cancelled`), and
hands each patch's body that `Progress` only for the detail phases that time
its renders under each sampler. Each patch's own phases and view counts go to
nothing, since dozens of rayon threads writing them into one bar would
overwrite each other.

The per-patch kernels of the other batches are different, and each has a plain
function beside a `_reporting` one that takes a `Progress`:
`refine_patch_normal`, `refine_patch_keypoints`, `fuse_patch_bitmap` and
`member_zncc_matrix`. None of them checks for cancellation or reports phases
or counts of its own; the `Progress` only times its renders under each sampler
in detail phases. The plain function is a single call of the `_reporting` one
with `Progress::none()`, so there is one body and no gate that could differ
between the two. The plain function serves the many callers with no
`Progress` to pass, most of them tests; the batch functions and the bench's
fit call the `_reporting` one.

Each view is rendered with the sampler the sampler rule picks for it
([image-warping.md](../camera/image-warping.md) § "Choosing the sampler per
view"): a track view at its keypoint, a candidate at its projection, as every
other kernel picks it for the same observation.

`projected_patch_frame` is the selector consumer's projection companion: it
returns an admitted view's centre pixel and projected `u`/`v` half-frame under
the same homogeneous finite/infinity convention. The bench geometry search
uses it to seed a candidate without carrying a second projection convention.

### The geometric cull

The pixel-free half of the candidate test is one public predicate:

```rust
/// Whether `camera` at `cam_from_world` could see `patch` at all: the patch
/// faces it, its centre is in front of it, and the patch's projected
/// footprint overlaps the image. Reads no pixels.
pub fn view_could_see_patch(
    patch: &OrientedPatch,
    camera: &CameraIntrinsics,
    cam_from_world: &RigidTransform,
) -> bool;
```

`select_patch_views` calls it for every candidate before it reads any pixel of
that view, so the selection and a caller that culls with it cannot drift apart.
A caller that fetches photographs can call it first and fetch only the views
that pass, putting a placeholder pyramid in every other slot: a view the
predicate rejects is never sampled. The viewer's geometry search does this, so
it reads the photographs of the track's own images and of the images that pass,
not of every image ([photograph-cache.md](../camera/photograph-cache.md) § "The
geometric cull").

The cull is **conservative**: it rejects a view only when the full selection
would also reject it, so it changes the time a search takes and never its
result. The front-facing and cheirality tests are the same code the selection
runs. The footprint test projects a 9 × 9 grid spanning the patch's `[-1, 1]`
extent, in the camera frame a render uses (so a point at infinity goes through
its homogeneous weight), widens the pixels' bounding box by the largest step
between neighbouring samples plus one pixel, and rejects only when that box
lies entirely outside `[0, width) × [0, height)`. If any sample fails to
project, the view is kept. A render whose every sample misses the frame fails
the selection's coverage test anyway. The `the_cull_*` tests and
`culled_views_as_placeholders_select_identically` in
[view_selection/tests.rs](../../../crates/sfmtool-core/src/patch/view_selection/tests.rs)
hold it to that.

### Affine candidate scoring (2026-07)

The candidate gate score exists only to admit/reject — nothing downstream
reuses the candidate render — so scoring does not need the full per-pixel
projective warp (previously ~86% of selection CPU, one `WarpMap::from_patch` +
`remap_bilinear` per candidate). Under every sampler a candidate is
scored through an **affine** patch→image map fit on its four exactly-projected
patch corners, sampling only the reference-support pixels at the affine
positions (same bilinear taps and `u8` rounding as `remap_bilinear`, so values
match wherever the positions do). Track-view diagnostic scores share the same
path.

A view takes this path whatever its sampler, so under the default sampler rule
both the views the rule leaves on `BilinearMip` and the views it moves to
`Anisotropic` do.

**Mip levels.** Under `BilinearMip` the affine map is additionally composed
with the pyramid level it minifies into, so the fast path reads the same level
the per-pixel path would. The map's linear part `∂(source px)/∂(grid px)` is
constant over the patch, so its larger singular value `σ_major` is evaluated
once and fed to the **same** level rule the per-pixel path applies
(`ℓ = round(log₂ max(σ_major, 1))`, clamped to the pyramid). Every coefficient
is then divided by `2^ℓ` — the level coordinate convention `x_ℓ = x_0 / 2^ℓ`
that the pyramid's 2×2 box downsample defines, with pixel centres at `+0.5` on
every level, so no half-pixel term enters — and the support is sampled from
level `ℓ` with one bilinear tap, which is exactly what `remap_bilinear_mip`
does per pixel. One level covers the whole patch where the per-pixel path may
straddle a boundary; that, and `σ_major` computed from the affine fit rather
than per-pixel central differences, are the two places the mip fast path can
differ from the slow one, and both fold into the same accepted
admission-flip loss below.

**The anisotropic footprint.** Under `Anisotropic` the affine map stays in
level-0 px. Its Jacobian is constant over the patch, so one SVD of it, with the
same `f32` arithmetic `WarpMap::compute_svd` applies per pixel (`svd_2x2`),
gives every support pixel the same footprint: the levels from `σ_minor` and
the samples along one major axis, the direction in the photograph the map
compresses most (the Jacobian's left singular vector). Each support pixel then takes the per-pixel
path's anisotropic sample (`remap::aniso_sample`, the body of the scalar
remap) at its affine position, rounded to `f32` as a warp map stores it
(`sample_support_affine_aniso`). On a map that is exactly affine the two paths
give the same bytes wherever the per-pixel SVD, read from finite differences of
the stored positions, lands on the same side of every level and sample-count
boundary as the constant one (`affine_aniso_sampling_matches_the_per_pixel_path`
holds seven such maps to bit identity, two of them a rotation composed with a
scale along one axis, whose Jacobian has perpendicular columns). Over stripes
that run along the compressed direction every sample of a walk reads the same
stripe, which `affine_aniso_walks_along_the_image_axis_the_map_compresses`
checks on maps where that direction is turned 0.8 to 1.57 rad from the grid
direction that compresses most. On a real view the affine position
error adds to that as it does for the bilinear samplers: on a long-focus view
of a square turned 70° the support samples differ from the exact anisotropic
render by at most 2 grey levels, 0.39 on average
(`affine_aniso_sampling_matches_exact_render_on_an_oblique_view`, held to the
bilinear pairs' bound of 4 and 0.8). The walk along the major axis can read up
to `σ_major / 2` source px past the quad; those taps are clamped at the frame
edge exactly as the per-pixel path clamps them, so the border gate is the
level-0 one.

The **exact warp remains the fallback** — and the sole authority on
rejection — whenever:

- a corner fails to project (behind camera / outside the model domain),
- the 4th-corner residual exceeds **0.5 px** (`AFFINE_MAX_RESIDUAL_PX` — the
  residual measures the **asymmetric** component of the projective +
  distortion curvature of the map over the patch, and bounds the interior
  position error for that class only; curvature symmetric about the patch
  centre — e.g. radial distortion around a patch near the principal point —
  cancels in the residual and folds into the accepted admission-flip loss
  instead. Large patches and wide-angle / heavily distorted views exceed the
  bound and fall back. The bound is in **source** px on every level: it
  measures the warp's curvature, and holding it in source px is the
  conservative reading when the samples come from a coarser level, where the
  same deviation is `2^ℓ` times smaller), or
- the mapped quad comes within ~1.5 px of the frame border
  (`AFFINE_BORDER_MARGIN_PX` — keeps every affine sample safely in-bounds and
  leaves the out-of-frame-support rejection semantics entirely to the exact
  path). The margin is applied in the **sampled level's** pixels, so the mip
  path declines a `2^ℓ`-px-wide border band — strictly more conservative than
  the level-0 path, which is the safe direction.

**Accepted loss:** admission flips for candidates near the ZNCC bar (the
affine position error perturbs scores by up to ~gradient × residual).
Measured on dino (85 imgs / 46k pts, identical inputs): mean per-point
admitted-set Jaccard vs the exact path **0.9943** (90.3% of points identical;
1.0% below 0.9), total admitted −0.079% (852 614 vs 853 291), flip rate 0.58%
of admissions. Cost: candidate scoring 49.2 → 25.5 µs/call, selection CPU
196 → 108 s, selection wall 6.1 → 3.4 s (70% of candidates take the fast
path; the rest are residual/border fallbacks). The remaining fast-path cost is
memory-latency-bound (scattered bilinear gathers over full-resolution
sources); a coarse-to-fine gate (low-resolution pre-score, full score only
near the admission bar) is the identified follow-up if selection needs to get
cheaper still.

Measured on kerry_park fisheye (OPENCV_FISHEYE, 320 imgs @ 960×960 / 12.9k
pts, identical inputs) — the regime where the residual's blind spot to
centre-symmetric distortion actually bites: **95%** of candidates take the
affine path (the radial-distortion curvature largely cancels in the 4th-corner
residual, so the gate does *not* divert them), and the admission agreement
degrades measurably but stays bounded: mean Jaccard **0.9833** (74.3%
identical sets, 0.43% of points below 0.8), total admitted −0.099%, flip rate
1.46%. This is the accepted symmetric-distortion loss described above, not a
safety issue (in-bounds sampling is residual-independent).

Both figures were measured with `Sampler::Bilinear`, before it stopped being
the default. The mip composition does not change what the fast path
approximates — the affine position error and the border/residual gates are
identical — so the admission-agreement figures carry over; what changes is the
cost it displaces, since the exact `BilinearMip` warp additionally computes the
warp map's per-pixel SVD. Measured per candidate-scoring call on the synthetic
selection fixture: `Bilinear` 13.0 → 3.7 µs (3.5×), `BilinearMip` 25.4 → 3.7 µs
(6.8×) — the fast path's own cost is level-independent (one bilinear tap per
support pixel either way, from a smaller and better-cached level).

### Implementation details

`PatchCloud.select_views(recon, images, *, min_relative_zncc=0.7, …,
min_self_agreement=0.3, point_indexes=None)`. The reference appearance is the
IRLS-weighted consensus of the track views' z-normalized patch renders over a
frozen common support (re-normalized per channel so a dot product is a windowed
ZNCC). A candidate is scored on the **reference's** surviving original channels
(a flat-in-the-candidate channel contributes 0), so the score is always a
correlation in one channel space — never the reference's channel A against a
candidate's channel B. Candidates are gated geometrically by
`view_could_see_patch`: `is_front_facing`, an explicit cheirality check (the
point must have positive camera-frame depth, since wide-fisheye / equirect
projection can map behind-camera points in-frame), and the footprint test. The track image indices are deduped order-preserving before use, so a
point with two observations in one image does not double-weight that view. The
self-agreement is the track views' mean ZNCC to the reference; when it is below
`min_self_agreement` (default 0.3) the track is admitted verbatim with no
expansion. The affine fast path covers every sampler: `Sampler::Bilinear`,
`Sampler::BilinearMip`, and `Sampler::Anisotropic`, whether the sampler rule
moved the view or the caller fixed that sampler. A point whose valid track-view count is below `min_track_views`
(default 2) likewise admits its track views verbatim. The render → z-normalize →
robust-consensus primitives are shared with `normal_refine` (`pub(super)`), not
duplicated.

## Limitations

- **No occlusion test.** Candidacy is geometric (front-facing, cheirality and
  footprint) and photometric; nothing compares a candidate's depth against the
  cloud, so on a non-convex or cluttered surface an occluded view is rejected
  only when its render disagrees with the reference.
- **No contrast floor.** "Not enough signal" is inferred from the
  self-agreement threshold. Self-agreement does not separate a textureless
  patch (little signal, untrustworthy) from a track whose views genuinely
  disagree (real signal, real disagreement); both fall below
  `min_self_agreement` and are admitted verbatim.
