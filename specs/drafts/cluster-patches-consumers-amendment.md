# Patch clusters in pair verification, patch embedding and the solver (amendment)

**Status:** Draft

Amends [`../core/patch/cluster-patches.md`](../core/patch/cluster-patches.md)
§ "Consumers", which lists what reads a cluster-patches file today and points
back here.

Three steps of the pipeline could use what cluster-patch refinement measures
and do not. Each is a direction to explore, not yet a design; none has a
measured case that asks for it.

## Photometric verification beside two-view geometry

`sfm match --derive-pairs` expands every member of every cluster into pairs and
then verifies the pairs with COLMAP's two-view geometry estimation. A member the
refinement kept has already passed a ZNCC gate and a shift gate against the
reference. Pairs drawn only from kept members could skip, or use looser
thresholds for, the descriptor-distance and geometric gates. What needs
measuring is how many pairs the photometric gate alone admits that two-view
geometry would reject, and whether those pairs are correct against a
ground-truth reconstruction.

## Seeding `embed-patches` frames from patch clusters

After a solve, `sfm embed-patches` builds each point's patch frame from the
point's tracks and their `.sift` keypoints. When a point's track came from a
patch cluster, the refined affine shape of each member already gives the
patch's scale and orientation in that view, so the frame could start from those
shapes instead of from the detections. The question is whether that start
reduces the work or the failures of the normal refinement that follows.

Per-cell displacements of each kept member, from which a frame and its normal
follow once poses exist, are measured by the piecewise refinement of
[cluster-patch-refinement.md](../core/patch/cluster-patch-refinement.md#piecewise-refinement)
and turned into a normal by
[cell-plane-normals.md](../core/patch/cell-plane-normals.md); a gate on that
normal and the seed's writer reading it are proposed in
[cell-plane-normal-precision-gate.md](cell-plane-normal-precision-gate.md).

## Clusters as track seeds for the solver

`sfm solve` reads pairs and refuses a cluster file. A solver that accepts tracks
directly could take the kept members of each patch cluster as one track,
instead of rebuilding tracks from the pairwise expansion. This depends on a
solver that takes tracks as input; the incremental and global mappers `solve`
runs build their tracks from the pairs in a COLMAP database.
