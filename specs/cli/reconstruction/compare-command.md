# `sfm compare` Command

## Overview

Compares two `.sfmr` reconstructions, reporting differences in alignment, camera intrinsics,
image poses, and feature usage.

## Command Syntax

```bash
sfm compare <RECONSTRUCTION1> <RECONSTRUCTION2>
             [--by-coordinate | --by-feature-index] [--pixel-threshold PX]
             [--fragments] [--fragment-pos-threshold PCT]
             [--fragment-rot-threshold DEG] [--fragment-min-size N]
             [--strips PATH [--strips-* ...]]
```

The `--strips*` family renders side-by-side patch-strip montages; see
"Side-by-side patch strips" below.

`RECONSTRUCTION1` is the reference; `RECONSTRUCTION2` is compared against it.

## Comparison Steps

1. **Alignment** — Computes a similarity transform (rotation, translation, scale) aligning
   the second reconstruction to the first from the shared images' camera poses.
2. **Camera intrinsics** — Compares focal length, principal point, and distortion parameters
   for matching cameras.
3. **Image poses** — For images present in both reconstructions, reports positional and
   angular differences after alignment. Position errors are reported relative to the reference
   **scene scale**: the root-mean-square distance of the reference's finite 3D points from
   their centroid. This makes the metric independent of each reconstruction's arbitrary SfM
   gauge, so a scene reconstructed 100× larger reports the same percentages. Rotation errors
   are in degrees and need no scaling.
4. **Fragment decomposition** — Decomposes the shared cameras into internally-rigid
   similarity components; see "Fragment decomposition" below.
5. **Feature usage** — Compares which features were triangulated in each reconstruction for
   shared images that use the same `.sift` file. Skipped (with a note) when either
   reconstruction stores no SIFT content hashes (e.g. an `embedded_patches` reconstruction).
6. **3D points** — Puts the two reconstructions' 3D points in correspondence and reports how
   many match / are unique to each side, plus the post-alignment distance distribution of
   the matched pairs (also reported relative to scene scale). Correspondences where either
   point is at infinity (homogeneous `w = 0`) are excluded from the distance statistics
   (their finite-vs-infinity distance is undefined) and reported separately. The overall
   relationship conclusion (`IDENTICAL` / `VERY SIMILAR` / `SIGNIFICANT DIFFERENCES`) uses
   scale-relative thresholds.

## Fragment decomposition

A single least-squares similarity fits the largest rigid subset of the shared cameras and
hides structural failures: a solve can consist of several internally-consistent **fragments**
at different scales and orientations, plus individually misplaced frames, while per-image
medians still look plausible. The decomposition surfaces both.

**Mechanism.** RANSAC over the not-yet-assigned shared cameras, repeated until exhaustion:

1. **Candidates** — every candidate similarity is estimated from a minimal subset of **two
   posed cameras**: the rotation comes from the camera orientations, the scale from the ratio
   of the center distances, the translation follows. All pairs are tried when there are few;
   otherwise a fixed-seed random sample of pairs (decomposition output is deterministic).
   Pairs with coincident centers (no scale constraint) are skipped.
2. **Consensus** — a camera supports a candidate when its position error is below
   `--fragment-pos-threshold` (default 3.5, as % of the reference scene scale — the same
   normalization the pose comparison uses) **and** its rotation error is below
   `--fragment-rot-threshold` (default 5 degrees).
3. **Extraction** — the largest consensus set is refit (a similarity estimated from all its
   members, re-collecting consensus until stable) and peeled off as the next component.
   The loop repeats on the remaining cameras until no consensus reaches
   `--fragment-min-size` (default 5) members.
4. **Outliers** — cameras left in no component are individual outlier frames.

**Report.** Components are listed largest-first with their camera count, first/last image
name, and internal position (% of scene scale) and rotation (degrees) error statistics under
their own transform. Every component after the first also reports its similarity **relative
to component 1**: the scale ratio, the rotation delta, and the **displacement** — the mean
distance between where component 1's transform and the component's own transform place its
cameras (% of scene scale). A raw translation is not reported: about the origin it would be
dominated by the scale mismatch. Outlier frames are listed by name (worst first) with their
position and rotation errors under component 1's transform. When the decomposition finds
more than one component or any outliers, the final conclusion adds a note.

**Echo verdict.** Every comparison prints a one-line echo verdict as a standard registration
check (whenever there are at least two shared cameras). A single rigid fragment is one
consistent gauge — reported as no echo. Otherwise it is an **echo**: part of the solve is
registered under a different similarity than the rest — a duplicated structure, as when a
capture returns to a viewpoint or a weakly-bridged arc is glued at the wrong relative pose.
The verdict quantifies it with the **largest fraction** (share of shared cameras in the
largest fragment) and the **echo offset** (median misplacement of the remaining cameras, % of
scene scale — secondary components by their displacement-vs-largest, outliers by their
position error). Both are available programmatically as `FragmentDecomposition.largest_fraction`
and `.echo_offset_pct`.

**Visibility.** The one-line echo verdict always prints. The detailed per-component/outlier
section is printed only when the decomposition finds something beyond a single all-inclusive
component — more than one component, or outlier frames — so a clean comparison's output stays
compact. `--fragments` forces the detailed section (including the single-component case).

## Point correspondence method

The 3D-point comparison (step 6) can correspond points two ways:

- **By feature index** (`--by-feature-index`) — keys observations on their feature index,
  which requires both reconstructions to reference the **same** `.sift` files (e.g. two solves
  over one workspace). Fast and exact, but yields no correspondences across different feature
  backends.
- **By keypoint coordinate** (`--by-coordinate`) — matches observations in shared images by
  *mutual nearest 2D keypoint* within `--pixel-threshold` pixels (default 2.0) and votes
  point-to-point. This works **across different SIFT backends** (e.g. COLMAP SIFT vs the
  sfmtool extractor), where the same scene keypoint lands at the same pixel under a different
  feature index. Keypoint coordinates are read from each reconstruction's workspace `.sift`
  files, or from the inline per-observation keypoints for an `embedded_patches`
  reconstruction (which has no `.sift` files).

The default is **auto**: coordinate matching is used when every shared image uses a different
SIFT file (including when either reconstruction stores no SIFT content hashes at all),
otherwise feature-index matching.

## Side-by-side patch strips (`--strips`)

`--strips OUT.png` renders points where the two solves differ as a side-by-side patch-strip
montage — one point per row, the reference solve's views on the left and the target solve's on
the right. Each tile is the point's oriented surfel projected into one observing view, so a
clean column means the solve placed a real, photoconsistent surface point there. The point
normals are refined for cross-view photoconsistency first (`--strips-no-refine` to skip).

> _**Precondition — left ungated (2026-06-25):** unlike `--refine-normals` and `render-patches`
> (which now require `embedded_patches`), `compare --strips` is **intentionally not gated**. It
> remains a dual-source diagnostic that builds patch clouds from raw solves on the fly: the strip
> montage is deeply `.sift`-tied (its frame sizing and ranking metrics read feature scales), and
> as a comparison/inspection tool it is most useful directly on solves without a conversion step.
> The diagnostic `_solve_strips` engine and the `scripts/exp_*`/`cmp_*` probes likewise keep
> calling the low-level `PatchCloud.from_reconstruction` directly._

The strip view always puts points in correspondence by keypoint coordinate (the robust choice
across SIFT backends), independent of the `--by-coordinate`/`--by-feature-index` choice used for
the numeric comparison above.

By default (`--strips-rank overview`) the montage shows a few points from each of several
categories, each under a labeled divider: least aligned, narrowest/widest triangulation angle,
most peripheral, largest NCC gap, lowest NCC, and points **unique** to each solve. A unique
point appears in one column with the other left blank. `--strips-rank` can instead focus on a
single axis (below).

Images and `.sift` files are resolved from each reconstruction's `workspace_dir` (as with the
photometric `xform` filters), so the workspaces must still hold them. Points at infinity
(`w = 0`) are skipped — they have no finite surfel to render.

Each row is annotated with the post-alignment distance (as % of scene scale), and per solve
the patch photoconsistency NCC and the point's triangulation angle (`a=…°`, small = depth
weakly constrained by parallax).

`--strips-rank` selects what to surface: either `overview` (default, the multi-category sampler
described above) or a single quantity, ranked toward one end (`--strips-end high|low`, default
the axis's natural end):

| `--strips-rank` | quantity | natural end (default) |
|---|---|---|
| `distance` | post-alignment distance between the two solves' placement | high (least aligned) |
| `view-angle` | triangulation angle | low (parallax-starved) |
| `ncc` | per-solve patch photoconsistency (min of the two solves) | low (suspicious surfel) |
| `ncc-gap` | gap between the two solves' photoconsistency | high (quality differs) |
| `image-radius` | keypoint distance from the principal point | high (peripheral) |
| `feature-size` | keypoint feature size in image pixels | high (coarse/blurry) |
| `world-size` | feature size projected to a metric surface footprint (size·depth/focal) | high (large patch) |

`ncc` and `ncc-gap` render every candidate to score it, so they are slower than the
geometry-only axes.

The two columns are named `reference`/`target` by default (the per-row `R`/`T` and the
`reference-only`/`target-only` labels follow); `--strips-labels LEFT,RIGHT` renames them (e.g.
`--strips-labels OLD,NEW`).

Other options: `--strips-num` (rows, default 16), `--strips-views` (cap tiles per strip,
default 8), `--strips-context N` (render wider NxN-pixel context tiles with a green box on the
validated extent; default 96, must exceed the 32px patch, 0 = off),
`--strips-refine/--strips-no-refine`.

## Usage Examples

```bash
# Compare two reconstructions
sfm compare sfmr/solve_001.sfmr sfmr/solve_002.sfmr

# Compare against ground truth
sfm compare ground_truth.sfmr sfmr/my_result.sfmr

# Always show the fragment decomposition, with a tighter consensus
sfm compare ground_truth.sfmr sfmr/my_result.sfmr --fragments --fragment-pos-threshold 2

# Force coordinate matching with a tighter threshold (e.g. comparing a COLMAP-SIFT
# solve against an sfmtool-SIFT solve of the same images)
sfm compare old.sfmr new.sfmr --by-coordinate --pixel-threshold 1.5

# Render the points the two solves place most differently as side-by-side patch
# strips, with wider scene context around each point
sfm compare old.sfmr new.sfmr --strips diff.png --strips-num 12 --strips-context 96
```
