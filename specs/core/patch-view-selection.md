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
| `min_relative_zncc` | ~0.7 | admit a candidate whose ZNCC to the reference clears this fraction of the track's own self-agreement |

## Implementation

Lives in `sfmtool-core::patch` (Rust), exposed through a PyO3 entry point. Reuses
the patch render + `is_front_facing` for candidacy and the IRLS consensus +
windowed ZNCC for the reference and scoring — the same machinery as normal
refinement and keypoint localization.

## Future work (not v1)

- **Occlusion-aware candidacy** — a geometric occlusion pre-filter (depth vs. the
  cloud) to bootstrap the vetting, hardening the non-convex / cluttered case.
