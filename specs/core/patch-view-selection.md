# Patch-View Selection

_Status: draft for review. The per-point **view-selection** algorithm used by the
[sift-based → patch-based reconstruction pipeline](sift-to-patch-reconstruction.md):
given a point with an oriented patch, it returns the views that *photometrically*
see that patch. It is a standalone, separately-callable algorithm, parallel to
[normal refinement](patch-normal-refinement.md) and
[keypoint localization](patch-keypoint-localization.md)._

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
   sees the surfel — the point projects in front of the camera and inside the
   frame, and the patch is front-facing (`OrientedPatch::is_front_facing`).
2. **Reference appearance.** Render the patch in each track view and combine them
   into a robust consensus — a reference image of what the surface looks like,
   from the views that already observe the point.
3. **Admit.** Render each candidate's patch (under the point's normal) and
   correlate it (windowed ZNCC) against the reference; admit those that clear a
   threshold tied to the track's own self-agreement (`min_relative_zncc`). Track views
   are always admitted.

This rejects self-occluded / disagreeing views that pure geometric visibility
would wrongly include.

## Output

The selected **view set** `G` — the admitted views for the point (track views
plus the photometrically-vetted candidates).

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

Lives in `sfmtool-core::patch` (Rust), exposed through a PyO3 entry point. Reuses
the patch render + `is_front_facing` for candidacy and the IRLS consensus +
windowed ZNCC for the reference and scoring — the same machinery as normal
refinement and keypoint localization.

### Affine candidate scoring (2026-07)

The candidate gate score exists only to admit/reject — nothing downstream
reuses the candidate render — so scoring does not need the full per-pixel
projective warp (previously ~86% of selection CPU, one `WarpMap::from_patch` +
`remap_bilinear` per candidate). With the bilinear sampler, a candidate is now
scored through an **affine** patch→image map fit on its four exactly-projected
patch corners, sampling only the reference-support pixels at the affine
positions (same bilinear taps and `u8` rounding as `remap_bilinear`, so values
match wherever the positions do). Track-view diagnostic scores share the same
path. The **exact warp remains the fallback** — and the sole authority on
rejection — whenever:

- a corner fails to project (behind camera / outside the model domain),
- the 4th-corner residual exceeds **0.5 px** (`AFFINE_MAX_RESIDUAL_PX` — the
  residual measures the **asymmetric** component of the projective +
  distortion curvature of the map over the patch, and bounds the interior
  position error for that class only; curvature symmetric about the patch
  centre — e.g. radial distortion around a patch near the principal point —
  cancels in the residual and folds into the accepted admission-flip loss
  instead. Large patches and wide-angle / heavily distorted views exceed the
  bound and fall back), or
- the mapped quad comes within ~1.5 px of the frame border
  (`AFFINE_BORDER_MARGIN_PX` — keeps every affine sample safely in-bounds and
  leaves the out-of-frame-support rejection semantics entirely to the exact
  path).

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

_Status: v1 implemented in `crates/sfmtool-core/src/patch/view_selection.rs`
(`select_patch_views` / `select_patch_cloud_views`), exposed as
`PatchCloud.select_views(recon, images, *, min_relative_zncc=0.7, …,
min_self_agreement=0.3, point_indexes=None)`. The reference appearance is the
IRLS-weighted consensus of the track views' z-normalized patch renders over a
frozen common support (re-normalized per channel so a dot product is a windowed
ZNCC). A candidate is scored on the **reference's** surviving original channels
(a flat-in-the-candidate channel contributes 0), so the score is always a
correlation in one channel space — never the reference's channel A against a
candidate's channel B. Candidates are gated geometrically by `is_front_facing`
**and** an explicit cheirality check (the point must have positive camera-frame
depth, since wide-fisheye / equirect projection can map behind-camera points
in-frame). The track image indices are deduped order-preserving before use, so a
point with two observations in one image does not double-weight that view. The
self-agreement is the track views' mean ZNCC to the reference; when it is below
`min_self_agreement` (default 0.3) the track is admitted verbatim with no
expansion. A point whose valid track-view count is below `min_track_views`
(default 2) likewise admits its track views verbatim. The render → z-normalize →
robust-consensus primitives are shared with `normal_refine` (widened to
`pub(super)`), not duplicated._

## Future work (not v1)

- **Occlusion-aware candidacy** — a geometric occlusion pre-filter (depth vs. the
  cloud) to bootstrap the vetting, hardening the non-convex / cluttered case.
- **Principled "insufficient data" gate** — a direct minimum windowed patch
  contrast / variance (SNR) test on the reference, rather than inferring
  "not enough signal" from the self-agreement threshold. Self-agreement conflates
  a textureless patch (low signal, untrustworthy) with a genuinely disagreeing
  track (real signal, real disagreement); an explicit contrast floor would
  separate the two and let the trust gate focus on disagreement.
