# Grown Plane Normal: Starting From a Textured Spot Off the Anchor

**Status:** Draft. Nothing here is built.

Amends: [core/bench/grown-plane-normal.md](../core/bench/grown-plane-normal.md)
§ "Non-goals" (a textured spot off the anchor) and § "Near to far within a
view".

## Purpose

`grown_plane_normal` grows every view's window from a 7×7 window centred on the
patch's anchor, the point where the reference keypoint's ray meets the patch.
The window is centred there because the shift each view's keypoint implies is
exact only at the anchor: every plane the step tries passes through the anchor,
so the right shift there does not depend on the plane, while at a spot `r` away
it differs by the plane error's displacement, about `(A − I)·r` for the error
`A − I` in the view's local affine map. A point whose centre lacks texture
therefore starts each view from a window that scores little and can mislead the
first fitted stage. The maintainer expects such points.

This draft proposes letting stage 1 start from a textured spot elsewhere in the
grid, compensating for the plane's displacement there.

## What is proposed

**Pinned-anchor tilt search.** Pin the anchor at the view's keypoint and search
the tilt at stage 1, so that a spot `r` from the anchor is matched at the shift
the tilt implies there rather than at the anchor's shift. The first window's
score then reads the tilt as well as the spot's texture, and the growing windows
start from the tilt the search found.

**Choosing the spot.** How the spot is chosen is open. The first round of the
Python prototype ranked the four non-overlapping 7×7 windows of the middle cell
by contrast, and the second took the one with the best stage-1 ZNCC among them
and the grid's centre (the multi-spot rule). On dino point 810 the spot decided
whether two wide views (images 65 and 66) converged from a start 22.8° off: the
most contrasted spot failed in both, ending about 12° off, while other spots
converged to 0.34° to 0.51°, and the spot with the best stage-1 ZNCC converged
in both. Those rules searched the first window's shift, which the step no longer
does; under the keypoint start, a spot off the anchor needs the compensation
above before any of them applies. On dino point 528, the prototype's
displacement at an off-centre spot put image 45's match more than 3 px from the
keypoint-implied shift, and its search locked onto a wrong shift 10 px away.

## Open questions

- Whether to build the pinned-anchor tilt search at all, and how the step would
  tell a point that needs it (its centre flat in the reference) from one that
  does not.
- How the spot is chosen, and whether the multi-spot rule's choice by stage-1
  ZNCC survives once the tilt is searched there.
- What the search's range and step are, and how its cost compares with the
  growth it saves.
