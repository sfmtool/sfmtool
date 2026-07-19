# Far-Field Rotation Initialization

**Status:** Specified — not implemented. Depends on `estimate_homography`
(specs/core/focal-vote.md), covisibility selection
(specs/core/covisibility-selection.md), rotation-locked resection
(specs/core/rotation-locked-resection.md), and the staged bundle
adjustment (specs/core/bundle-adjustment.md).

## Purpose

Build an initial multi-camera reconstruction from cluster tracks by using
the two point populations for what each observes: parallax-free
(far-field) correspondences fix rotations between arbitrary image pairs
through conjugate homographies `H = K R K⁻¹`, independent of baseline;
parallax-bearing (near-field) correspondences then supply the metric
side — a seed baseline, structure, and translation growth — with
rotations held. The output is a posed core (rotations, translations,
triangulated points) for a caller's refinement machinery; it succeeds
precisely on captures whose windowed parallax is too weak for
factorization-style seeding.

## Inputs

The flat cluster-observation arrays (as in `focal_vote`), the shared
image size, a focal `f0` (typically a focal-vote consensus), and a seed.

## Mechanism

### 1. Rotation edge graph

Candidate pairs per image: the largest-mean-displacement covisible
partners (displacement tables from covisibility selection; at least 25
shared clusters, displacement at least `0.05 × diagonal`, up to 3 edges
per image). Per candidate: estimate the homography over the pair's
shared-cluster correspondences (centred coordinates); require at least
12 inliers; validate as a conjugate rotation at `f0` by the
orthogonality residual (`< 0.12`; a finite-plane homography never
passes). A validated edge stores `R_ij` — the polar-orthogonalized
`K⁻¹ H K` — and its inlier partition: H-inliers are the edge's far
field, H-outliers its near field.

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
observing at least 12 triangulated points resects its translation;
after each growth round, retriangulate all clusters over the posed set
and repeat until no image is added or the core reaches its size budget
(`max_images`, default 14). Finish with one staged bundle adjustment
(full default schedule) over the posed set at fixed `f0`.

## Output

Posed-image indices with rotations (WXYZ) and translations, the
triangulated points (`NaN` where absent), each posed image's surviving
inlier fraction from the final adjustment, and the per-edge far-field
cluster ids (callers may feed these to the bundle adjustment's
points-at-infinity mask).

## Binding

```python
rotation_init(cluster_indexes, image_indexes, positions_xy,
              width, height, f0, seed=0,
              min_images=8, max_images=14)
    -> {"image_indexes", "quaternions_wxyz", "translations",
        "points", "inlier_fractions", "far_cluster_indexes"} | None
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
