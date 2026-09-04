# Far-Field Rotation Initialization

## Purpose

Far-field rotation initialization poses a first handful of cameras on
captures whose parallax is too weak to seed any other way, by letting the
distant, parallax-free correspondences fix the rotations first and the near
ones supply the metric frame afterwards. It builds that initial
multi-camera reconstruction from cluster tracks by using the two point
populations for what each observes: parallax-free (far-field)
correspondences fix rotations between arbitrary image pairs through
conjugate homographies `H = K R K⁻¹`, independent of baseline;
parallax-bearing (near-field) correspondences then supply the metric
side — a seed baseline, structure, and translation growth — with
rotations held. The output is a posed core (rotations, translations,
triangulated points) for a caller's refinement machinery.

## Inputs

The flat cluster-observation arrays (as in `focal_vote`), the shared
image size, a focal `f0` (typically a focal-vote consensus), and a seed.

## Frame convention

Every rotation this kernel takes in or hands back is in the **canonical camera
frame**: the camera looks along `−Z`, with `+Y` up, which is the frame the
`.sfmr` format stores and the rest of the codebase works in. The returned
`quaternions_wxyz` and `translations` are world-to-camera in that frame.

The conjugate homography does not live there. `H = K R K⁻¹` is a relation
between *pixel* coordinates, and the pixel frame is the optical one — `+X`
right, `+Y` **down**, looking along `+Z` — so the rotation that comes out of
`K⁻¹ H K` is expressed optically. The two frames differ by a flip of the last
two axes, so the conversion is a conjugation by

```
S = diag(1, −1, −1),      R_canonical = S · R_optical · S
```

which is its own inverse (`S² = I`). This is applied once, at the boundary where
an edge is built: nothing downstream of `build_edges` sees an optical-frame
rotation, and nothing upstream of it sees a canonical one. Getting it wrong is
silent — `S R S` is still a rotation matrix, of the same angle — so it shows up
only as a reconstruction that is mirrored about the horizontal axis.

## Mechanism

### 1. Rotation edge graph

One pass over the cluster runs builds the shared-cluster count and mean
feature displacement of every covisible image pair, exactly as the focal
vote's pair tables do: each cluster keeps one position per member image
(last observation wins) and contributes every pair of its distinct member
images. The kernel builds these tables itself rather than reading the
sampled `ClusterCovisibility` tables, because a single sampled member pair
per cluster undercounts covisibility far enough to starve the 25-shared
gate that follows, on exactly the parallax-poor captures this method
exists for.

Candidate pairs per image: the largest-mean-displacement covisible
partners (at least 25 shared clusters, displacement at least
`0.05 × diagonal`, up to 3 edges per image). Per candidate: estimate the
homography over the pair's shared-cluster correspondences (centred
coordinates); require at least 12 inliers; validate as a conjugate
rotation at `f0` by the orthogonality residual (`< 0.12`; a finite-plane
homography never passes). A validated edge stores `R_ij` — the
polar-orthogonalized `K⁻¹ H K`, conjugated by `S` into the canonical frame (see
[Frame convention](#frame-convention)) — and its inlier partition: H-inliers are
the edge's far field, H-outliers its near field.

### 2. Global rotations

Over the largest connected component of the edge graph (fail below
`min_images`, default 8): spanning-tree propagation from the
highest-degree image, then iterative rotation averaging to consensus —
each image's rotation is re-estimated as the chordal mean of its
neighbours' propagated estimates (`R_j ← mean over edges (i,j) of
R_ij · R_i`), sweeping until the largest single-image update falls below
0.1° or 20 sweeps. Averaging exists to absorb tree drift: a chain of
edge rotations accumulates error that a tree alone passes to its leaves.

### 3. Seed baseline and structure

The component edge with the most near-field correspondences seeds the
metric frame: with `R` known on both ends, the epipolar constraint is
linear in the translation direction (`x₂ · (t × R_rel x₁) = 0`, solved
by SVD over the near rows); the sign is fixed by triangulation
cheirality (majority in-front wins, minimum 10). The seed pair's
near-field clusters triangulate into the initial structure; the second
camera's translation defines unit scale.

### 4. Translation growth

Grow over the component by rotation-locked resection: any unposed image
observing at least 12 triangulated points is a candidate, and its
translation resects under an 8 px trim gate needing at least 10
survivors. After each growth round, retriangulate all clusters over the
posed set and repeat until no image is added or the core reaches its size
budget (`max_images`, default 14). Finish with one staged bundle
adjustment (full default schedule) over the posed set at fixed `f0`.

That adjustment models the far field at infinity, over a mask the kernel
builds itself as the deduplicated union of the H-inlier clusters of the
component's validated edges. The mask is not optional and not the
caller's to supply: with the far clusters left as finite points, a
dominant far cloud rewards baseline collapse, and the LM walks the flat
scale gauge downward until the near field crosses the adjustment's trim
depth floor and the core degenerates to a panorama — each staged round is
individually well behaved, and the walk compounds across them. Because
the gauge is flat it can also wander harmlessly, so after the adjustment
the posed translations and the finite points are rescaled to pin the seed
baseline back to unit; the far rows are directions and are left alone.

## Output

Posed-image indices with rotations (WXYZ) and translations, the
triangulated points (`NaN` where absent, and unit world-frame directions
on the far-field rows the finishing adjustment modeled at infinity), and
each posed image's surviving inlier fraction from the final adjustment.
The far-field mask itself is internal: a caller that needs to know which
clusters were held at infinity reads the unit rows off `points`.

## Binding

`rotation_init` lives in
[rotation_init.rs](../../../crates/sfmtool-core/src/geometry/rotation_init.rs),
bound as `sfmtool._sfmtool.geometry.rotation_init` by
[rotation_init.rs](../../../crates/sfmtool-py/src/geometry/rotation_init.rs).
It builds on `estimate_homography` ([focal-vote.md](focal-vote.md)),
rotation-locked resection
([rotation-locked-resection.md](rotation-locked-resection.md)), and the
staged bundle adjustment ([bundle-adjustment.md](bundle-adjustment.md)).

```python
rotation_init(cluster_indexes, image_indexes, positions_xy,
              width, height, f0, *, seed=0,
              min_images=8, max_images=14)
    -> {"image_indexes", "quaternions_wxyz", "translations",
        "points", "inlier_fractions"} | None
```

## Testing requirements

- Synthetic far-field-rich scene (near cloud + distant cloud, known
  poses): rotations recovered to sub-degree after averaging; averaging
  measurably beats tree-only propagation on a long chain with per-edge
  noise.
- Seed and growth: recovered translations and structure match ground
  truth up to similarity on synthetic data; growth stops at
  `max_images`; a component below `min_images` returns `None`.
- A capture with no valid rotation edges (all-parallax scene, every
  homography rejected by the orthogonality floor) returns `None`.
- Determinism under a fixed seed; binding parity.

## Non-goals

- Focal estimation (`focal_vote` owns it; `f0` is an input).
- Growing beyond the core budget or verifying against appearance —
  widening and photometric verification belong to the caller.
- Loop-closure detection beyond what rotation averaging over the edge
  graph already provides.
