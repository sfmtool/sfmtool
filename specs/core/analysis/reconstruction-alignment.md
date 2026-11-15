# Reconstruction alignment: least-squares similarity fit + RANSAC outlier rejection

## Overview

Aligning one reconstruction to another means estimating a **similarity
transform** (rotation `R`, translation `t`, uniform scale `s`, together an
`Se3Transform`) that maps source-frame 3D points onto their target-frame
counterparts. The point-based alignment pipeline is:

1. Find corresponding 3D point pairs between the two reconstructions via
   their observations in shared images (paired as
   [Pairing the images](#pairing-the-images) says), by feature index or by keypoint
   position — see
   [point-correspondence.md](../reconstruction/point-correspondence.md). Pairs involving a
   point at infinity are dropped up front (a `w = 0` point stores a bearing
   direction, not a metric location, and would corrupt the fit).
2. Optionally run RANSAC over the correspondences to reject outlier pairs
   (mismatched or badly-triangulated points).
3. Fit the final similarity transform to the surviving inliers with the
   least-squares fit.

Camera-based alignment (`sfm align --method cameras`) does **not** use this
module — it estimates the similarity from matched camera poses in pure
Python (`align/core.py::estimate_similarity_with_orientations`, weighted
quaternion averaging per Markley et al.). That is a *second* implementation
of reconstruction-level alignment, parallel to `reconstructions.rs` below and
answering the same question differently; retargeting `sfm align` at the
shared Rust one is open work.

## Pairing the images (`image_pairing.rs`)

Every alignment of two reconstructions starts by deciding which image of one
is which image of the other. The name, the image's path relative to its
workspace, decides most of it, but a folder copied into another workspace
layout, or frames renamed after extraction, give a photograph another name.
The hash of the image file pairs those, so images are paired by that hash
first and then by name. A name pairs whatever the hashes say: two
reconstructions of the same photographs at another resolution, or
re-encoded, have different files under the same names, and they pair by name.
`sfm align`, `sfm compare`, `sfm xform --align-to` and the SfM Explorer's
`Align to` all pair images this way.

The pairing is in
[image_pairing.rs](../../../crates/sfmtool-core/src/analysis/alignment/image_pairing.rs):

```rust
pub struct ImageIdentity<'a> { pub name: &'a str, pub file_hash: Option<[u8; 16]> }

pub struct ImagePairing {
    pub pairs: Vec<(u32, u32)>,                    // (source, target), by source index
    pub by_file_hash: usize,
    pub by_name: usize,
    pub by_name_different_files: Vec<(u32, u32)>, // a subset of pairs
}

pub fn pair_images(source: &[ImageIdentity], target: &[ImageIdentity]) -> ImagePairing;
pub fn pair_reconstruction_images(
    source: &SfmrReconstruction,
    target: &SfmrReconstruction,
    progress: &Progress,
) -> Result<ImagePairing, Cancelled>;
```

`pair_images` takes names and hashes rather than reconstructions so that a
caller pairing many reconstructions reads each one's hashes once (`sfm align`
pairs every two of its inputs). `pair_reconstruction_images` reads both
reconstructions' hashes and pairs them; `align_reconstructions` calls it. Python
reaches the pairing as `sfmtool.analysis.pair_images(source_names,
source_file_hashes, target_names, target_file_hashes)`, which returns a dict of
the same fields as `uint32` `(k, 2)` arrays and counts, plus the `summary`
line.

```rust
let pairing = pair_reconstruction_images(&source, &target, &Progress::none())?;
println!("{}", pairing.summary()); // images paired: 15 by file hash, 2 by name
```

**The rule**, in order:

1. **By file hash.** Images whose file hashes are known and equal pair,
   whatever their names. A hash held by exactly one image on each side pairs
   those two. A hash held by more than one image on a side is the same file
   under several names; among the images holding it, only images of the same
   name pair, the first image of a name on each side, and an image of that
   group with no same-named partner in the other side's group is left to
   step 2. These pairs are counted in `by_file_hash`.
2. **By name.** Each image still unpaired pairs with the still-unpaired image
   of the same name on the other side, whether or not their hashes are known
   or equal. A repeated name pairs once, its first unpaired image on each
   side. These pairs are counted in `by_name`.

Every pair that pairing by name alone (the first image of a name on each side)
makes is therefore still made, unless one of its two images was paired in
step 1 with another image of the same file. A name pair is only ever given up
for a pair by file hash.

Beside the counts, `by_name_different_files` lists the pairs made by name
whose two file hashes are both known and differ: the same photograph at
another resolution or re-encoded, or the frames of two videos, which share a
name and are different photographs. Pairing cannot tell these apart, so the
list is information for a reader and changes no pair.
`ImagePairing::summary` reads `images paired: 15 by file hash, 2 by name`,
followed by `, 2 of them different files` when the list is not empty. The CLI
and the viewer print this line as it is.

**Where the hashes come from.**
`SfmrReconstruction::find_image_file_hashes(progress)` returns one
`Option<[u8; 16]>` per image, the XXH128 of the image file's bytes:

- `embedded_patches`: the stored `image_file_hashes` column, so every hash is
  known.
- `sift_files`: the `image_file_xxh128` that image's `.sift` file records in
  its metadata, read from the feature folder the `.sfmr` records
  (`sift_path_for_image`). Only the file's metadata is read; no photograph is
  read or hashed. An image's hash is `None` when its `.sift` file is missing or
  cannot be read, when the file's content hash is not the image's
  `sift_content_hashes` entry (another extraction, perhaps of another
  photograph), or when its `image_file_xxh128` is not a hash; every hash is
  `None` for a reconstruction with no workspace directory. None of these is an
  error: the image pairs by name.

Reading a `sift_files` reconstruction's hashes opens one `.sift` file per
image, which is why it takes a `Progress`. It reads only what the given `.sfmr`
records, which opening that file may always do, so it needs no version 2
workspace ([workspace.md](../../workspace/workspace.md#a-command-that-uses-a-workspace-requires-version-2)).
Python reaches it as `SfmrReconstruction.find_image_file_hashes(progress=None)`,
a list of 16-byte `bytes` or `None`.

## Aligning two reconstructions (`reconstructions.rs`)

The alignment core lives in
[least_squares.rs](../../../crates/sfmtool-core/src/analysis/alignment/least_squares.rs),
[ransac.rs](../../../crates/sfmtool-core/src/analysis/alignment/ransac.rs) and
[reconstructions.rs](../../../crates/sfmtool-core/src/analysis/alignment/reconstructions.rs),
with the two point-set estimators bound as
`sfmtool.analysis.estimate_alignment` and `ransac_alignment`.
Python drives it from `sfm align --method points`
([by_points.py](../../../src/sfmtool/align/by_points.py)) and the `sfm xform`
operations `--align-to` / `--align-to-input`
([_align_to.py](../../../src/sfmtool/xform/_align_to.py),
[_align_to_input.py](../../../src/sfmtool/xform/_align_to_input.py)).

The two estimators above fit *point sets*. `align_reconstructions(source,
target, options, progress) -> Result<AlignFit, String>` fits *reconstructions*: it
decides which of the source's geometry corresponds to which of the target's,
drives the estimators over the result, and reports how well it went. This is
the shared implementation behind the SfM Explorer's `Align to ▸` (see
[scene-graph.md](../../gui/scene-graph.md), "Node Transforms and
Alignment") and the entry point any other Rust caller aligning two loaded
`SfmrReconstruction`s should use.

`AlignOptions` is the whole choice: `source` (`AlignSource::Cameras` — the
default — or `AlignSource::Points`) and `estimate_scale` (similarity vs
rigid). The numeric policy is fixed in the module rather than exposed:
`AlignmentParams { rounds: 3, keep_fraction: 0.8 }` for the trimmed refit,
and for the point mode a 200-round RANSAC pass with sample size 3, seed 42,
and a threshold taken as the 95th percentile of a preliminary
all-correspondence fit's residuals (floored at `1e-9 ×` the target cloud's
extent, so an exactly-corresponding pair is not rejected as all-outlier by
`f64` rounding noise).

Both modes start from `pair_reconstruction_images` ([Pairing the
images](#pairing-the-images)); `progress` is passed to it, the one part of the
fit that reads files for every image, and a cancelled pairing fails the fit.

**Correspondences by cameras** — the paired images' camera *centres* feed the
fit. Fewer than 3 shared images is refused rather than
fitted — two points leave the rotation about the line joining them free. No
RANSAC: a few hundred centres are too few for a consensus sample to beat
fitting them all and trimming the worst away.

**Correspondences by points** — observations in the shared images matched
into 3D point pairs (see
[point-correspondence.md](../reconstruction/point-correspondence.md)). When both
sides have feature indexes (`sift_files`), `find_point_correspondences` matches
them by feature index. Otherwise `find_point_correspondences_by_coordinate`
matches them by keypoint position with its default options (2 px, 2 votes).
Each side's positions come from `observation_pixels`: the inline
`keypoints_xy` column when the reconstruction carries one (always for
`embedded_patches`), else the `.sift` positions its feature indexes name, read
from the workspace for the shared images only. A `.sift` file that cannot be
read fails the fit with a reason naming the side and the file. Pairs touching
a point at infinity are dropped afterwards, as in `sfm align`. Fewer than 10
correspondences, before or after RANSAC, is refused (`sfm align`'s
`min_points`).

`AlignFit` carries the fitted `transform` (source's native coordinates →
target's native coordinates; composing it into a frame either reconstruction
is *displayed* in is the caller's job), the correspondence count, the inlier
count, the RMS residual over the inliers, which mode produced them, and, for
the point mode, `point_matching`: `PointMatching::FeatureIndex` or
`PointMatching::KeypointPosition` (`None` for the camera mode), and `images`,
the `ImagePairing` the fit started from. Every
failure the modes name — no shared images, too few correspondences, a
degenerate configuration — comes back as an `Err` string a caller can show
verbatim, leaving the source where it was. The two that are about the images
end with the pairing's summary in parentheses: `the two reconstructions share
no images (images paired: 0 by file hash, 0 by name)`.

## Least-squares fit (`least_squares.rs`)

`estimate_alignment(source_points, target_points, n_points, params) ->
Result<Se3Transform, AlignmentError>` computes the least-squares similarity aligning
`n` uniformly-weighted source points to target points. Points are flat
`&[f64]` slices of length `3 * n` (row-major `x0, y0, z0, x1, …`). `params`
is an `AlignmentParams { rounds, keep_fraction, estimate_scale }`; its
`Default` (`rounds = 1`, `keep_fraction = 1.0`, `estimate_scale = true`) is the
single-shot similarity fit described here — see [Trimming and rigid
fits](#trimming-and-rigid-fits) for the other knobs.

The classical SVD construction (the Kabsch/Umeyama solution):

1. **Centroids** of both sets; both sets are centered.
2. **Cross-covariance** `H = Σ w · (sᵢ − s̄)(tᵢ − t̄)ᵀ` with uniform
   `w = 1/n`.
3. **SVD** `H = U Σ Vᵀ`, then `R = V Uᵀ`.
4. **Proper-rotation correction:** if `det(R) < 0`, the third row of `Vᵀ` is
   negated before recomputing `R` (reflection → rotation).
5. **Scale** `s = Σ w · tᵢ' · (R sᵢ') / Σ w · sᵢ' · sᵢ'` (primes are the
   centered points). When the source points have zero variance (single point
   or all coincident), scale is undetermined and defaults to `1.0`.
6. **Translation** `t = t̄ − s · R · s̄`.

### Rank-deficient input

For degenerate configurations the SVD's nullspace columns are arbitrary
orthonormal completions, so the raw `V Uᵀ` could contain an arbitrary
rotation about the unconstrained axes. The implementation fixes only the
genuinely unconstrained cases, keyed off the numerical rank of `H`
(singular values relative to the largest, threshold `1e-10`):

- **Rank 0** (single point / all coincident): `R = I`.
- **Rank 1** (collinear points): the two nullspace columns of `V` are forced
  to match `U`'s, so the rotation about the constrained axis is identity.
- **Rank 2+**: fully determined by SVD + the det correction (the third axis
  is fixed by orthonormality); left untouched.

### Errors

`AlignmentError` is shared with `ransac_alignment` below.
`estimate_alignment` returns `AlignmentError::NoPoints` when `n_points` is 0
and `AlignmentError::PointsLength` when either slice does not hold
`n_points * 3` values; the binding raises each as `ValueError`. Note the Rust
core accepts `n = 1` (yielding `R = I`, `s = 1`, pure translation); the Python
wrapper `align/core.py::estimate_alignment` demands `n ≥ 2`.

### Trimming and rigid fits

`params.rounds > 1` enables trimmed refitting: after the first fit over all
correspondences, each subsequent round re-selects the `keep_fraction` of
correspondences with the smallest residual `‖s·R·src + t − tgt‖` under the
previous fit and refits on that subset, discarding gross mismatches.
`params.estimate_scale = false` fixes the scale at `1.0` (a rigid fit) by
skipping the scale step. With the default `rounds = 1` the trimming loop does
not run, so the result is identical to the single-shot fit above.

## RANSAC (`ransac.rs`)

`ransac_alignment(source_points, target_points, n_points, max_iterations,
threshold, min_sample_size, seed) -> Result<Vec<bool>, AlignmentError>`
returns a per-correspondence inlier mask. It returns
`AlignmentError::PointsLength` when either slice does not hold
`n_points * 3` values and `AlignmentError::SampleSize` when
`min_sample_size` is 0 or more than `n_points`; the binding raises each as
`ValueError`. Standard hypothesize-and-verify:

1. Sample `min_sample_size` indices without replacement (seeded `StdRng` —
   deterministic for a given `seed`).
2. Fit a similarity to the sample with `estimate_alignment`; degenerate
   samples (fit error) are skipped.
3. Transform **all** source points; a pair is an inlier when the Euclidean
   distance to its target is `< threshold`.
4. Keep the mask with the highest inlier count over `max_iterations`.

If no iteration produces a single inlier (or every sample is degenerate),
the initial all-`true` mask is returned — i.e. "no consensus found" degrades
to "keep everything" rather than "reject everything".

## Python surface and CLI parameterization

Binding defaults (`analysis/core.rs`): `max_iterations=1000`,
`threshold=0.1`, `min_sample_size=3`, `seed=42`.

`sfm align --method points`
(`align/by_points.py::estimate_alignment_from_points`) wires it up as:

- **Correspondences** come from the same finder choice as
  `align_reconstructions`: `find_point_correspondences` (by feature index)
  when both sides have feature indexes, `find_point_correspondences_by_coordinate`
  (by keypoint position, 2 px, 2 votes) when either is `embedded_patches`.
  The `sfm xform` `--align-to` operations share this function.
- **Several targets in one frame:** the function takes a list of
  `(target reconstruction, image pairs)` rows, all targets in the same
  coordinate frame. Correspondences are found with each target separately,
  since each target's pairs index its own images, then concatenated, and
  the threshold, RANSAC and final fit below run over the union. `sfm align`
  passes every aligned neighbour of the input being aligned; `--align-to`
  passes its one target. A target whose finder raises
  `NoPointCorrespondencesError` (a `ValueError` raised only when no finite
  correspondence is found) adds none; when every target's finder raises it,
  their messages are raised together as one `ValueError`. Any other error,
  such as a `.sift` file that does not match its reconstruction, fails the
  alignment.
- **Threshold is data-derived, not fixed:** a preliminary least-squares fit
  over *all* correspondences is computed, and the threshold is the
  `--ransac-percentile` (default 95.0) percentile of its residual
  distances.
- `--ransac-iterations` (default 1000) → `max_iterations`; `seed` is fixed
  at 42 for reproducibility; `min_sample_size` stays at the binding default 3.
- RANSAC is skipped when `--no-ransac` is given or when the correspondence
  count is not above `min_points` (default 10). Fewer than `min_points`
  correspondences — before or after RANSAC — is an error.
- The final transform is a least-squares re-fit over the inliers only; the reported
  RMS error and confidence are computed from the inlier residuals.

See [`specs/cli/reconstruction/align-command.md`](../../cli/reconstruction/align-command.md) for the CLI
surface and multi-reconstruction ordering, and
[point-correspondence.md](../reconstruction/point-correspondence.md) for how the input pairs
are found.

## Testing

Sibling `tests.rs` files under `analysis/alignment/least_squares/` and
`analysis/alignment/ransac/` cover exact recovery of known transforms,
rank-deficient inputs (identical points, collinear points), reflection
avoidance, scale recovery, and RANSAC inlier/outlier separation with
deterministic seeds.

`analysis/alignment/image_pairing/tests.rs` covers each step of the pairing
rule: images of one file paired across different names, the same name on two
files left unpaired and reported, a name pairing where a hash is missing, an
image paired by hash not paired again by name, a hash held by several images
on one or both sides, a repeated name, and the summary.
`reconstruction/image_file_hashes/tests.rs` covers where the hashes come from:
the stored column, a `.sift` file whose content hash matches, one that is
another extraction, one with no hash, a missing file, no workspace, and a
cancelled read.

`analysis/alignment/reconstructions/tests.rs` works one level up, on a fixture
pair built the way the GUI spec asks for: reconstruction B is reconstruction A
put through a *known* similarity, so the assertions are about where B's points
and cameras land rather than about the transform's components (which sidesteps
the quaternion sign ambiguity). It covers both modes' round trips, the point
mode by feature index and by keypoint position (an `embedded_patches` pair, a
mixed pair, and the checked-in Seoul bull ground truth against a moved copy of
itself), that the landed cameras face the way the target's do, rigid vs
similarity, the exclusion of points at infinity, renamed images of the same
files aligning by their hashes in both modes, same-named images of different
files not being paired, and each refusal path.
