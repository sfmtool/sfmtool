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

- The flat cluster-observation arrays, as in `focal_vote`: per observation a
  cluster id (`cluster_indexes`), an image id (`image_indexes`) and a
  full-pixel keypoint position (`positions_xy`), all of one length.
  `cluster_indexes` must be nondecreasing, so that each distinct cluster is
  one contiguous run. The Rust kernel does not check this, and an unsorted
  array splits a cluster into several runs with no error; the binding raises
  `ValueError` on it.
- The image size `width × height`, shared by every image. It sets the
  principal point, at the image centre, and the image diagonal that the
  displacement gate in § 1 scales by.
- A focal `f0` in pixels, shared by every image, typically a focal-vote
  consensus.
- A `seed` for the homography RANSAC, and the core's size bounds
  `min_images` and `max_images`.

The kernel models every image as one distortion-free pinhole camera, a
`SimplePinhole` at `f0` with its principal point at the image centre, and
reads the positions as pixels of that camera. It applies no distortion model
anywhere: the homography test of § 1, the seed rays and the resection all
assume that camera. Positions from a lens with real distortion, such as a fisheye,
give wrong poses and raise no error, so a caller with such a camera
undistorts the positions to a pinhole first.

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
rotation, and nothing upstream of it sees a canonical one. Getting it wrong raises
no error, because `S R S` is still a rotation matrix, of the same angle. Since
`S = −M` with `M = diag(−1, 1, 1)`, `S R S = M R M`: an unconjugated rotation
is the rotation of the scene mirrored left to right, so rotations about the
vertical `Y` axis and the viewing `Z` axis come out in the wrong sense and
rotations about `X` do not. The observations are not mirrored, so the result
is not a clean mirror image of the scene but starting rotations that disagree
with the observations. When those rotations are small, the finishing
adjustment of § 4 can move them back: the synthetic scenes in
[tests.rs](../../../crates/sfmtool-core/src/geometry/rotation_init/tests.rs)
pan by about 2° per image and still pass with the conjugation removed.

## Mechanism

### 1. Rotation edge graph

One pass over the cluster runs builds the shared-cluster count and mean
feature displacement of every covisible image pair, exactly as the focal
vote's pair tables do: each cluster keeps one position per member image
(last observation wins) and contributes every pair of its distinct member
images. The kernel builds these tables itself rather than reading the
sampled `ClusterCovisibility` tables, because a single sampled member pair
per cluster undercounts covisibility far enough to leave too few pairs
with the 25 shared clusters the gate that follows requires, on exactly the
parallax-poor captures this method exists for.

Candidate pairs: each image proposes its 3 covisible partners of largest
mean displacement among those with at least 25 shared clusters and a mean
displacement of at least `0.05 × diagonal` (`MAX_EDGES_PER_IMAGE = 3`). The
cap is on what an image proposes, not on its edges: an image also gains every
pair a partner proposes, so it can end with more than 3 edges, and a
candidate that fails validation is not replaced by the next partner. Each
pair is tried once. Per candidate: estimate the homography over the pair's
shared-cluster correspondences, in coordinates centred on the principal
point, with
[`estimate_homography`](focal-vote.md#homography-estimation) at an inlier
gate of 3 px of symmetric transfer error (`H_MAX_ERROR_PX = 3.0`); require
at least 12 inliers; validate as a conjugate
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
0.1° or 20 sweeps. Averaging reduces the error that accumulates along a
chain of edge rotations, which tree propagation alone passes on to the
images at the end of the chain.

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

That adjustment starts the far field at infinity, over a mask the kernel
builds itself as the deduplicated union of the H-inlier clusters of the
component's validated edges. The mask is not optional and not the
caller's to supply: with the far clusters started as finite points, a
dominant far cloud rewards baseline collapse, and LM reduces the scale
toward a zero baseline, where the core degenerates to a panorama. Each staged
round is individually well behaved, and the reductions add up across the
rounds. Started as
directions, the far clusters sit at zero inverse depth, where they carry no
translation column. The adjustment runs with the default crossing
([bundle-adjustment.md](bundle-adjustment.md) § "Free points: inverse depth
and the storage decision"), so at its end every point, far-field or not, is
stored as a position or a direction by what its rays support at the noise the
adjustment measures. Because the scale gauge is flat, the scale can also change without affecting
the cost, so after the adjustment the posed translations and the positions are rescaled
to pin the seed baseline back to unit; the directions are left alone.

## Output

Posed-image indices with rotations (WXYZ) and translations, the
triangulated points (`NaN` where absent), `point_at_infinity` saying which
rows are unit world-frame directions as the finishing adjustment stored them,
and each posed image's surviving inlier fraction from the final adjustment.
The far-field mask the adjustment starts from is internal.

The kernel returns `None` when:

- the arrays are empty, or their lengths differ (the binding raises
  `ValueError` on differing lengths instead);
- `f0` is not finite and positive, or `max_images < 2`;
- no rotation edge validates;
- the largest connected component of the edge graph has fewer than
  `min_images` images, or fewer than 2;
- the seed edge has fewer than 10 near-field correspondences, or its
  near rows give no usable translation direction;
- fewer than 10 of the seed's near-field correspondences triangulate in
  front of both cameras under either sign of the translation.

## Binding

`rotation_init` lives in
[rotation_init.rs](../../../crates/sfmtool-core/src/geometry/rotation_init.rs),
bound as `sfmtool.geometry.rotation_init` by
[rotation_init.rs](../../../crates/sfmtool-py/src/geometry/rotation_init.rs).
It builds on `estimate_homography` ([focal-vote.md](focal-vote.md)),
rotation-locked resection
([rotation-locked-resection.md](rotation-locked-resection.md)), and the
staged bundle adjustment ([bundle-adjustment.md](bundle-adjustment.md)).

```rust
// sfmtool_core::geometry::rotation_init
pub fn rotation_init(
    cluster_indexes: &[u32],
    image_indexes: &[u32],
    positions_xy: &[[f64; 2]],
    width: u32,
    height: u32,
    f0: f64,
    seed: u64,
    min_images: usize,     // DEFAULT_MIN_IMAGES = 8
    max_images: usize,     // DEFAULT_MAX_IMAGES = 14
) -> Option<RotationInit>
```

```python
rotation_init(cluster_indexes, image_indexes, positions_xy,
              width, height, f0, *, seed=0,
              min_images=8, max_images=14)
    -> {"image_indexes", "quaternions_wxyz", "translations",
        "points", "point_at_infinity", "inlier_fractions"} | None
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
- The rows `point_at_infinity` marks are unit rows, and at least 80% of them
  are far-cloud clusters.
- Determinism under a fixed seed; binding parity.

## Non-goals

- Focal estimation (`focal_vote` owns it; `f0` is an input).
- Growing beyond the core budget or verifying against appearance —
  widening and photometric verification belong to the caller.
- Loop-closure detection beyond what rotation averaging over the edge
  graph already provides.
