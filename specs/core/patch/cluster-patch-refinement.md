# Cluster-Patch Refinement

## Purpose

A feature matcher can decide that a handful of detections spread across a
handful of images probably came from the same point on a surface, but it says
almost nothing about how that piece of surface *looks* from each of those
images. Cluster-patch refinement fills that in, before any camera pose exists.
Given one such group of detections and the pictures they came from, it elects
one member as the group's reference, cuts a small square of image around it,
and then, for every other member, searches for the affine transform of that
square which best reproduces what the member's image actually shows. What comes
back, per member, is a photometrically verified local shape, a corrected
keypoint position, and a verdict on whether the member belongs at all.

Doing this early pays twice: a later stage can read a member's image-space
extent and refined position directly, and members that agree on a descriptor
while disagreeing on appearance become visible without a reconstruction.

An optional second stage, the **piecewise refinement**, follows the affine fit
for every kept member. It cuts the patch into nine cells, registers each cell
separately against the member's photograph at the shape the affine fit found,
and stores where each cell's content lies relative to that shape. A normal
needs camera poses and cannot be computed at this stage, but these per-cell
displacements are the pose-free half of one: once poses exist,
[cell-plane-normals.md](cell-plane-normals.md) turns them into a patch normal
without rendering anything. The stage is off by default, and when it runs it
leaves every other output of the refinement unchanged.

This document specifies the kernel: `sfmtool-core`'s `patch::cluster_refine`,
its PyO3 binding, and the numerics. The kernel is pure — no I/O, no `.sift`
reads. The caller hands it decoded image pyramids, the SIFT geometry of every
image, and the clusters in CSR form; it hands back member-parallel arrays that
map 1:1 onto the `cluster_patches/` section. The motivation, the rejected
alternatives and the measured calibration behind the defaults live in
[cluster-patches.md](cluster-patches.md); the on-disk sections are specified
normatively in
[matches-file-format.md](../../formats/matches-file-format.md); the command is
[`sfm cluster-patches`](../../cli/image-feature/cluster-patches-command.md).

## Rust API

The kernel lives in
[cluster_refine](../../../crates/sfmtool-core/src/patch/cluster_refine/mod.rs),
with its parameters and result in
[params.rs](../../../crates/sfmtool-core/src/patch/cluster_refine/params.rs) and
the piecewise stage in
[piecewise.rs](../../../crates/sfmtool-core/src/patch/cluster_refine/piecewise.rs),
and is bound as `sfmtool._sfmtool.matching.refine_cluster_patches` in
[cluster.rs](../../../crates/sfmtool-py/src/matching/cluster.rs). The stored
per-cell columns are `MemberCellData` and `ClusterCellStatus` in
[cells.rs](../../../crates/sfmtool-matches-format/src/cells.rs).

```rust
/// Per-member verdict. Discriminants match `sfmtool_matches_format::ClusterMemberStatus`.
#[repr(u8)]
pub enum MemberStatus {
    Reference = 0,
    Kept = 1,
    RejectedLowZncc = 2,
    RejectedShift = 3,
    DuplicateImage = 4,
    NotEvaluated = 5,
    RejectedUnlocalizable = 6,
}

/// `reference_members` entry for a cluster with no usable reference.
pub const REFERENCE_UNREFINABLE: u32 = u32::MAX;

pub struct ClusterRefineParams { /* … see Parameters */ }

/// One image's SIFT feature geometry (borrowed views of the `.sift` arrays).
pub struct FeatureGeometry<'a> {
    pub positions_xy: ArrayView2<'a, f32>,   // (N, 2), source px
    pub affine_shapes: ArrayView3<'a, f32>,  // (N, 2, 2), keypoint frame → px
}

pub struct ClusterRefineResult {
    pub reference_members: Vec<u32>,          // (C,)
    pub member_status: Vec<MemberStatus>,     // (M,)
    pub member_positions: Array2<f64>,        // (M, 2) = p
    pub member_affine_shapes: Array3<f64>,    // (M, 2, 2) = S
    pub member_zncc: Vec<f32>,                // (M,), NaN if not evaluated
    pub member_zncc_middle: Vec<f32>,         // (M,), the same over the middle square
    pub member_zncc_grid: Vec<[[f32; 3]; 3]>, // (M,), the same over each ninth of the grid
    pub member_shift_px: Vec<f32>,            // (M,), NaN if not evaluated
    pub cells: Vec<Option<CellRefinement>>,   // (M,), Some only for a kept member
                                              // when `params.piecewise` is set
}

pub fn refine_cluster_patches(
    pyramids: &[ImageU8Pyramid],
    features: &[FeatureGeometry<'_>],   // parallel to `pyramids`
    cluster_starts: &[u32],             // (C+1,) CSR
    member_images: &[u32],              // (M,)
    member_features: &[u32],            // (M,)
    params: &ClusterRefineParams,
    progress: Option<&AtomicUsize>,     // one tick per finished cluster
) -> ClusterRefineResult;

/// One member's own `R×R` grid at a position and affine shape: what the
/// member gate reads.
pub fn sample_member_grid(
    pyramid: &ImageU8Pyramid, position: [f64; 2], affine_shape: [[f64; 2]; 2],
    params: &ClusterRefineParams,
) -> Option<Vec<f32>>;
/// The number the member gate judges: the overlap reading of that grid.
pub fn member_zncc_self_similarity_radius(
    pyramid: &ImageU8Pyramid, position: [f64; 2], affine_shape: [[f64; 2]; 2],
    params: &ClusterRefineParams,
) -> Option<f64>;

impl ClusterRefineParams {
    pub fn member_self_similarity_gate_is_on(&self) -> bool;
    pub fn admits_member_zncc_self_similarity_radius(&self, radius: f64) -> bool;
}

/// The reconstruction-free contamination signal computed from the refined
/// warps; see [cluster-warp-consistency.md](cluster-warp-consistency.md).
pub fn warp_consistency_residuals(
    cluster_starts: &[u32],
    member_images: &[u32],
    member_status: &[MemberStatus],
    reference_members: &[u32],
    member_affine_shapes: ArrayView3<'_, f64>,  // (M, 2, 2)
    n_images: usize,
) -> Vec<f32>;
```

**Why this shape.** Pyramids and borrowed feature views rather than file paths,
because decoding and `.sift` reading belong to the orchestration layer that
already holds both — the separation `ProjectedImage` gives the other patch
kernels. Three flat CSR arrays in and member-parallel vectors out, rather than
anything nested, because that is the shape the clusters have on disk, in the
matcher's in-memory cluster set and in numpy; nesting would be repacked at every
boundary. Nothing returns a `Result` — a member that cannot be evaluated is
*data* (`NotEvaluated`, `RejectedUnlocalizable`, …), and the failures that
remain are caller bugs (non-parallel inputs, malformed CSR), which assert.
`warp_consistency_residuals` stands apart from the result because it is a fit
over a whole refined cluster set, and a caller that only wants warps should not
pay for it. The member-grid sampler and the radius are public so the bench
([editable-track.md](../bench/editable-track.md)) reads a cluster-stage
sighting's self-similarity from the same tile the gate reads, rather than from a
second copy of the sampling.

```rust
use sfmtool_core::camera::image::{ImageU8, ImageU8Pyramid};
use sfmtool_core::patch::cluster_refine::{
    refine_cluster_patches, ClusterRefineParams, FeatureGeometry, MemberStatus,
};

let pyramids: Vec<ImageU8Pyramid> =
    images.iter().map(|im: &ImageU8| ImageU8Pyramid::build(im, 5)).collect();
// One entry per image, borrowing that image's (N, 2) and (N, 2, 2) f32 `.sift`
// arrays.
let features: Vec<FeatureGeometry<'_>> = sift
    .iter()
    .map(|s| FeatureGeometry {
        positions_xy: s.positions.view(),
        affine_shapes: s.shapes.view(),
    })
    .collect();

let out = refine_cluster_patches(
    &pyramids,
    &features,
    &[0u32, 3, 5],           // two clusters: members 0..3 and 3..5
    &[0u32, 1, 2, 0, 3],     // member images
    &[17u32, 42, 8, 91, 5],  // member features
    &ClusterRefineParams::default(),
    None,
);

// Member 1's image-space extent and refined position, if it survived vetting.
if out.member_status[1] == MemberStatus::Kept {
    let s = out.member_affine_shapes.index_axis(ndarray::Axis(0), 1);
    let extent = (s[[0, 0]].hypot(s[[1, 0]]), s[[0, 1]].hypot(s[[1, 1]]));
    let p = out.member_positions.index_axis(ndarray::Axis(0), 1);
    let position = [p[0], p[1]];
    let _ = (extent, position);
}
```

### The piecewise stage

`ClusterRefineParams::piecewise: Option<PiecewiseParams>` turns the stage on;
`None`, the default, leaves `cells` all `None`.

```rust
pub struct PiecewiseParams {
    pub move_shape: bool,          // let the fitted map move the shape (the loop)
    pub cell_shift_bound_px: f32,  // shift search bound, and the render margin
    pub min_cell_zncc: f32,        // a cell below this at its optimum is refused
    pub min_cell_curvature: f32,   // a flatter ZNCC peak is refused
    pub update_tolerance_px: f32,  // with `move_shape`: convergence, grid px
    pub max_iterations: u8,        // with `move_shape`: render cap
}

/// How far the whole-member ZNCC may fall at an updated shape before the
/// update is rejected (with `move_shape`).
pub const ACCEPT_ZNCC_TOLERANCE: f64 = 1e-4;
pub const DEFAULT_MIN_CELL_ZNCC: f32 = 0.8;
pub const DEFAULT_MIN_CELL_CURVATURE: f32 = 0.02;

/// Discriminants match `sfmtool_matches_format::ClusterCellStatus`.
#[repr(u8)]
pub enum CellStatus {
    Fitted = 0,
    RefusedCurvature = 1,
    RefusedZncc = 2,
    NotAttempted = 3,
    RefusedBound = 4,
    RefusedOutlier = 5,
}

#[repr(u8)]
pub enum LoopStop { NotRun = 0, Converged = 1, Cap = 2, Rejected = 3, Oscillation = 4, Measured = 5 }

/// One kept member's cells, `[row][col]` from the top-left cell.
pub struct CellRefinement {
    /// Displacement of each cell's centre from where the returned shape
    /// places it, template grid px, with no fitted affine map removed.
    pub shift_px: [[[f32; 2]; 3]; 3],
    pub zncc: [[f32; 3]; 3],
    pub status: [[CellStatus; 3]; 3],
    pub iterations: u8,              // renders: 1 without `move_shape`
    pub stop: LoopStop,              // `Measured` without `move_shape`
    pub final_update_accepted: bool, // always false without `move_shape`
}

/// The cells as the format's four member-parallel columns: `NaN` / `NaN` /
/// not attempted / `0` for a member without cells.
pub fn member_cell_data(cells: &[Option<CellRefinement>]) -> MemberCellData;
```

**Why this shape.** The displacements are relative to the member's affine
shape, not absolute, so a consumer that ignores them reads the file exactly as
before. They are stored as measured, with no fitted affine map removed: the
stored shape and the displacements together then say where each cell's content
lies, and a consumer that wants only the part no affine map matches fits and
removes the affine part itself, whereas one that wants the whole displacement
could not recover what a stored residual had dropped. They are in template grid
px, the unit every other refined quantity in the file uses. The status is per
cell because the point of the stage is to know which cells to trust; one
per-member flag would discard that. `cells` is one entry per member, `None` for
a member that is not kept, so it indexes like every other per-member array and a
consumer cannot pair a member with another member's cells. The shift bound and
the render margin are one parameter because they must be equal: a search that
can reach a shift the render did not cover reads outside the tile. `stop` and
`final_update_accepted` are present in both modes, reading `Measured` and
`false` without `move_shape`, so the result has one shape. `CellStatus` mirrors
the format's enum for the same reason `MemberStatus` does: the binding writes
the codes straight into the file, and a test checks the discriminants agree.

**Example.**

```rust
use sfmtool_core::patch::cluster_refine::{
    member_cell_data, refine_cluster_patches, CellStatus, ClusterRefineParams,
    PiecewiseParams,
};

let params = ClusterRefineParams {
    piecewise: Some(PiecewiseParams::default()), // measure, never move the shape
    ..ClusterRefineParams::default()
};
let out = refine_cluster_patches(&pyramids, &features, &starts, &images, &feats, &params, None);
if let Some(cells) = &out.cells[1] {
    let middle = cells.shift_px[1][1];          // grid px from the stored shape
    let trusted = cells.status[1][1] == CellStatus::Fitted;
    let _ = (middle, trusted);
}
let columns = member_cell_data(&out.cells);     // what the `.matches` writer stores
```

## Theory

### The template

Every SIFT detection carries a position and a 2×2 affine shape `A` mapping the
detector's canonical *keypoint frame* onto image pixels. The reference's patch
is therefore a square in keypoint-frame coordinates: `resolution²` samples at
pixel-center offsets spanning `[−radius, radius]` per axis
(`step = 2·radius / resolution`, offset `u = k·step + 0.5·step − radius`),
carried into the image by `x = pos_ref + A_ref · u`. At the default `radius = 6`
the template spans 12 keypoint-frame units per axis — SIFT's own ~12×
descriptor window, so it vets a member against roughly the texture the detector
judged characteristic of the feature.

Only the *windowed* support is sampled. The scoring window is normal
refinement's shared `PatchWindow`, whose sigma is in normalized patch
coordinates where the grid spans `[−1, 1]²`; the default
`GaussianDisk { sigma: 0.5 }` is a Gaussian of half the patch half-width
confined to the inscribed disk, so in-plane rotation is exactly free and grazing
corners cannot leak in. Each channel is sampled independently, z-normalized over
that window, and kept only if its windowed norm clears `FLAT_NORM_SQ_EPS` — a
channel flat under the window carries no information. If **any** support sample
falls outside the image, or every channel is flat, the candidate reference is
unusable and the next candidate is tried.

### The objective

The score is the window-weighted zero-mean normalized cross correlation of the
z-normalized template against the member image sampled through the current warp,
per surviving channel, averaged over the template's channel count. Channel
identity is preserved — the reference's channel *c* is only ever correlated
against the member's channel *c*, and a member with fewer channels contributes
nothing for the missing ones. A member channel flat under the window scores `0`
for that channel rather than a garbage ratio.

Sampling is all-or-nothing: if any support sample leaves the frame the
evaluation returns the worst possible score (`+1.0` on the negated objective)
rather than a partial correlation, so the optimizer retreats instead of being
rewarded for walking off the image. Partial support would make scores from
different warps incomparable — exactly the ordering the optimizer relies on.

### The warp and the cascade

The unknown is an affine map from the reference's patch onto the member's image,
seeded from the detectors themselves (`M₀ = A_mem · A_ref⁻¹`, anchored at the
two detections):

```
W(x) = pos_mem + t + (I + D) · M₀ · (x − pos_ref)
```

`t` is a translation in source pixels, `D` a 2×2 correction to the seed's linear
part. The family stops at affine because the calibration measured the true
patch-to-patch warp at SIFT-patch scale as affine to well under a hundredth of a
pixel on every dataset, fisheye included: perspective terms only overfit.

The search is a three-stage Nelder-Mead cascade, each stage seeded from the
previous stage's optimum:

| stage | parameters | seed |
|---|---|---|
| shift | `t` (2) | `t = 0` |
| similarity | `t, σ, φ` (4), `D = e^σ R(φ) − I`, `σ` clamped to ±1.5 | the shift optimum |
| affine | `t, D` (6) | the similarity optimum, `σ`/`φ` expanded into `D` |

Each simplex is seeded at `θ₀ + scale_i·e_i` with 0.5 px for translations and
0.05 for shape entries, under standard coefficients (reflect 1, expand 2,
contract 0.5, shrink 0.5). Starting at translation matters: the detections'
*positions* are the noisiest part of the seed, and letting the shape float
before the patch is centred spends evaluations chasing a mis-registered
template. The shift and similarity stages exist only to seed the affine stage,
so they stop on a looser tolerance (`intermediate_convergence`) than the stage
whose answer is stored (`convergence`). Every stage additionally stops on a
stall — no improvement of the best value by more than `stall_tol` for
`stall_iters` consecutive iterations. That exit is for the affine stage, whose
reflect-heavy 6-dim crawl on a flat objective shrinks its value *spread* far
more slowly than it stops making *progress*; without it most members ran to the
iteration cap long after the score stopped moving. There is no multi-view
congealing pass — at raw-cluster sizes it measurably adds nothing over pairwise
refinement.

### Pyramid levels

For every sampled image — the template's and each evaluation's — the level is
`ℓ = clamp(⌊log₂ s_min⌋, 0, L−1)`, where `s_min` is the smaller singular value
of the support map's linear part (the sample spacing in source pixels along the
compressed axis); the map is divided by `2^ℓ` before sampling. The level is
chosen **per objective evaluation**, not once per member, because `D` changes
the linear part as the cascade runs. Sampling from too fine a level aliases the
score surface the optimizer descends, a worse failure than the blur of a
slightly coarse one. Full anisotropic footprints would be more correct still and
are deliberately not used: this is a single tap from the selected level, the same
choice the other patch kernels make, differing only in the level rule
(`floor(log₂ s_min)` here against `round(log₂ σ_major)` there).

### Which member anchors, and which members are eligible

Before anything is refined, each member's own patch is read for its
[ZNCC self-similarity radius](zncc-self-similarity-radius.md): how far, in
template-grid px, the member's full `resolution²` grid at its own SIFT geometry
can slide over itself and still match itself as well as a true match between two
views would. The reading uses the default `SelfSimilarityParams` (shifts up to
`max_radius = 3` px) on the member grid itself (`sample_member_grid`), read
[the overlap way](zncc-self-similarity-radius.md#the-overlap-reading): at each
shift only the samples both windows hold are correlated, so no pixel outside the
grid enters the reading. A member whose
radius is above `max_member_zncc_self_similarity_radius` becomes
`RejectedUnlocalizable` and takes no further part: a flat wash or a straight
edge can neither anchor a cluster nor honestly join one, since it matches itself
along the edge or everywhere and so agrees photometrically with a translation
the refinement cannot pin. The gate samples with a nearest-valid-pixel clamp
rather than an in-frame requirement, so a member near the border is read on its
visible content instead of escaping the gate; only non-finite geometry (or a
pyramid level too small to sample) skips it. The pass rule is the keypoint
localizer's ([patch-keypoint-localization.md](patch-keypoint-localization.md)):
a member passes when its radius is at or below the bar, a `NaN` radius fails, a
bar of `0` (or a non-finite one) disables the gate, and since the radius reads
at most `3`, a bar of `3` or more turns nothing out. The default, `2.5`, is the
localizer's `DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS`, so the two member
gates start from the same bar.

The reference is then the surviving member with the largest SIFT scale
`√|det A|`, ties to the lowest global member index — a larger patch resolves the
smaller ones rather than the reverse. Selection is policy, not format: the
reference is stored as data, so a better policy can replace this one without a
format change. If the best candidate's template proves unusable the next
candidate by scale is tried, and a cluster where every candidate fails is
*unrefinable*: `reference_members[c] = REFERENCE_UNREFINABLE`, and every member
not already gated is `NotEvaluated`. The same follows when fewer than two
members survive validation, which also drops members with an out-of-range
feature index or a degenerate shape (`|det A| < 1e-9`).

### Acceptance

A refined member is vetted in a fixed order: ZNCC below `min_zncc` →
`RejectedLowZncc`; otherwise `|t|` above `max_shift_px` → `RejectedShift`;
otherwise kept. The ZNCC gate is deliberately permissive, because the measured
failure mode is over-culling rather than contamination and because the achieved
ZNCC routinely exceeds the *ground-truth* warp's own — the score gates match
validity, never warp correctness. Consumers re-gate on the stored signals, which
is why rejected members keep their measured ZNCC and shift.

**The middle ZNCC.** Beside each member's ZNCC the result carries
`member_zncc_middle`: the member's support samples at the final map, the one the
winning evaluation used, correlated against the reference template's own raw
samples over only the middle of the grid, the centred square half its width
(rows and columns `R/4 .. R - R/4`). Each channel is mean-removed and normalized
over that square alone, under the same window weights and the same channel
pairing as the whole-patch score. It costs one more pass over the tile the
refinement already built for that map, and no gate reads it. A match the middle
does not share is carried by the parts of the patch away from its centre: the
background behind a small near object, the far side of a depth edge, or a
texture that repeats along the epipolar line. The reference reads `1` against
itself unless its middle is flat, and every member reads `NaN` where the
reference's middle is flat or where the member was not evaluated. The
`.matches` cluster-patches section does not store it.

**The ZNCC grid.** The member is also read at the same final map over the
whole `R×R` square, not only the support, and correlated against the
reference's own samples over the same square, which the template samples
beside its support. The result is `member_zncc_grid`, one ZNCC per cell of a
three-by-three split of the square, the rows and columns cut at `R/3` and
`R - R/3`, `[row][col]` from the top-left cell. Each cell is mean-removed and
normalized alone, as the middle is, with every pixel weighted equally, so the
corners the window's disk leaves out are read in full. It is read in the same
pass as the middle, and no gate reads it. The reference reads `1` in every cell
it is not flat over. A cell is `NaN` where the reference is flat over it or a
pixel of it falls off the image, and every cell is `NaN` where the member was
not evaluated. The `.matches` section does not store it either.

Finally, at most one member per image survives: among provisionally kept members
sharing an image the highest ZNCC wins, ties to the lowest member index, and the
rest become `DuplicateImage` — as does any member sharing the reference's own
image, which is marked before it is ever refined.

### What is returned, and why it is absolute

The working unknown is the relative warp `W = (I + D)·M₀`, but the result
reports the **absolute affine shape** `S = W · A_ref` and the member's
**refined absolute keypoint position** `p = pos_mem + t`. `S` is literally the
matrix the winning evaluation sampled with, so the reported shape is the shape
that was measured, and because it maps the detector's canonical unit frame onto
that member's image pixels, its column norms are the member's image-space
extent: a consumer reads extent and position per member with no `.sift` file
open. The reference member's own entries are `A_ref` and its detected position,
so the relative warp stays recoverable as `W = S · S_ref⁻¹`, after which
`x_mem = W·(x − x_ref) + p`. That inversion is what keeps a *derived* file — one
whose reference member has been filtered out — meaningful: absolute shapes and
positions stay valid regardless, and only the relative reading needs the
reference.

The two arrays are member-parallel but not member-complete: a member the
cascade never fitted (`NotEvaluated`, `RejectedUnlocalizable`, and a
`DuplicateImage` that shared the reference's image) has an all-zero entry, and
`member_status` is what says so. The `.matches` writer reads that: it stores
the refinement's values for the measured members and leaves the input's
detections in place for the rest, so the file it writes has no holes (see
[`matches-file-format.md`](../../formats/matches-file-format.md), Member
geometry). Nothing about the member's geometry is stored in `cluster_patches/`, which
carries the vetting evidence and, when the piecewise stage ran, the per-cell
readings.

### After refinement

`warp_consistency_residuals` factors all of a cluster's recovered warps against
one weak-perspective camera per image and one tangent frame per cluster, and
reports each member's misfit — a reconstruction-free contamination signal that
catches the wrong-match member which aligns photometrically on repetitive
texture ([cluster-warp-consistency.md](cluster-warp-consistency.md)). It is
computed in the same binding call and stored beside the ZNCC and shift, as a
signal and not a gate. Once tracks and poses exist, the sharper pairwise
agreement test of
[member-coherence-validation.md](member-coherence-validation.md) decides
membership on the same kind of evidence with a reconstruction in hand.


### Piecewise refinement

The affine shape says how the surface around a feature appears in one view as a
whole: its scale, orientation and skew. It cannot say that half the patch lies
on a different surface, or how the surface's appearance varies across the patch.
The piecewise refinement measures that. Its output is a 2D displacement for each
of nine cells of each kept member, which once poses exist becomes a sighting ray
per cell; the rays of one cell across the members meet at the cell's position in
the world, and a plane through the nine positions is the patch normal
([cell-plane-normals.md](cell-plane-normals.md)).

#### The cells

The cells are the three-by-three split of the template's `R × R` grid that the
ZNCC grid uses: rows and columns cut at `R/3` and `R − R/3`, `[row][col]` from
the top-left cell. A cell's centre is the mean of its grid positions
(`grid_cell_centres`, shared with the normal refinement and with
[cell-plane-normals.md](cell-plane-normals.md)). The template is the cluster's
reference member, so the displacements are relative to that member's frame and
the reference's own cells displace by zero by construction; only kept members
are measured, since a member that is not kept has no trustworthy affine shape to
start from.

#### The measurement

A member's cells are measured at the shape the affine cascade found for it. The
stage reads the photograph once, and everything after that reads only a small
tile.

1. **Render.** The member's affine shape maps the template's grid into the
   photograph, and the photograph is resampled along that map into a **working
   patch**: the template's `R × R` grid plus a margin of `cell_shift_bound_px`
   on every side (rounded up). This is the only step that reads the photograph.
2. **Register the cells.** Each of the nine cells of the template is compared
   with the working patch by windowed ZNCC at every whole shift within the
   bound (rounded down). The margin guarantees that every shift of every cell,
   edge cells included, reads pixels the render covered. At the best whole
   shift a sub-pixel peak is fitted to the three-by-three neighbourhood of
   ZNCC values, and the peak's curvature, the smaller eigenvalue of the negated
   Hessian, says how sharply the cell is pinned.
3. **Gate each cell.** A cell is refused as `refused_curvature` when the
   reference is flat over it or its peak is flatter than `min_cell_curvature`:
   a cell over flat or repeating texture registers anywhere, and its ZNCC may
   still be high because a flat region correlates with a flat region at every
   shift. It is refused as `refused_zncc` when its ZNCC at the optimum is below
   `min_cell_zncc`, that is when it is over a different surface. It is refused
   as `refused_bound` when its best whole shift lies on the search bound, since
   no whole-pixel neighbour on that side was read and its optimum is at or past
   the bound. A cell whose search could not read a sample it needs is
   `not_attempted`.
4. **Fit an affine map.** The surviving cells' centres and shifts are point
   correspondences between the template and the working patch, and a robust
   affine map is fitted through them (below). A cell the fit gives no weight is
   stored as `refused_outlier`, with its shift; every other survivor is
   `fitted`. The map decides which cells agree with each other. Without
   `move_shape` it is not stored and does not move the shape.

At `patch_size = 12` a cell is 4×4 template px: enough for a shift registration
against a textured template, and not enough for a self-similarity ellipse, which
is why the per-cell gate reads the peak's curvature rather than the
[ZNCC self-similarity radius](zncc-self-similarity-radius.md). The search is a
shift search only: the cascade has removed scale and rotation, and a cell that
needs more than a shift is over a different surface and should be refused, not
fitted.

#### The robust affine fit

A rule on the count and the spread of the surviving cells chooses the model: a
full affine with five or more well-spread cells, a similarity with three or
more otherwise, a pure shift below that, and no fit when none survive, in which
case all nine cells are stored as `not_attempted`. "Otherwise" covers three or
four cells, five or more that lie along one row or column and so pin only one
axis of an affine, and five or more well-spread cells whose affine normal
equations are singular: each falls back one step, to the similarity, before the
shift.

A cell can pass both of its own gates and still be wrong: a repeated texture can
give it a sharp, high-scoring peak at the wrong shift, and a depth edge inside it
can give it a peak that belongs to the other surface. Such a cell disagrees with
the map the others agree on. The map is therefore fitted by iteratively
reweighted least squares: a first fit weighted by each cell's curvature, then
three rounds in which each cell's weight is its curvature times a Tukey biweight
of its residual to the previous round's fit. The residuals are two-dimensional
lengths, so the residual scale is the median residual length divided by
`√(2 ln 2)`, about 0.849 times it: the per-axis standard deviation of isotropic
Gaussian noise whose residual lengths have that median. The factor `1.4826`
used for one-dimensional residuals would overstate it by a factor of 1.75. The
scale is floored at 0.1 grid px, so that cells agreeing to a few hundredths of a
pixel do not make a cell a tenth of a pixel off an outlier, and the biweight's
cut-off is 4.685 scales. A cell at or past the cut-off has weight zero and is
`refused_outlier`. The model choice reads the cells with a positive weight.

#### What is stored, and what it carries

The stored displacement of a cell is the displacement of its centre from where
the member's stored shape places it, with no fitted affine map removed. Without
`move_shape` that is the shift as measured, and the member's shape, position,
whole-member ZNCC, its parts and its shift from the seed are the cascade's, bit
for bit. The displacements carry three things:

- **A first-order part.** The cascade found the affine shape that maximises one
  windowed ZNCC over the whole patch; each cell's optimum is found separately on
  far fewer samples, so the nine optima do not agree exactly with the
  whole-patch optimum. The part of that disagreement one affine map can express
  is in the displacements. On the fleet subset the fitted cells' displacements
  have a median length of 0.16 to 0.35 grid px, most of it this part.
- **The second-order term.** For a planar surface, the part of the cell shifts
  no affine map matches is the second-order term of the homography, which is
  what a normal needs.
- **Parallax.** For a surface that is not one plane, the cells off the plane the
  others agree on are displaced by their parallax, which is large and is what
  refuses those cells.

A consumer that wants only the second-order term fits an affine map to the
displacements of the `fitted` cells and removes it. The file stores four
member-parallel entries, `member_cell_shift_px`, `member_cell_zncc`,
`member_cell_status` with its legend, and `member_cell_iterations`, specified in
[matches-file-format.md](../../formats/matches-file-format.md#per-cell-entries-optional-version-8).
They are optional entries of `.matches` version 8; version 9 adds
`refused_outlier` to the statuses the legend may name. A member that is not kept
carries `NaN` / `NaN` / `not_attempted` / `0`. Why the loop stopped and whether
its last update was applied are in the result and the binding only.

The cell shifts are measured on a tile that is itself an interpolation of the
photograph. For a shift read as a sub-pixel ZNCC peak this is a small blur, not a
bias.

#### Why the shape is left to the cascade

The cell fit's affine map is an update the cells agree on, and could move the
shape. But the two objectives disagree: the cell fit asks for the map that best
explains nine separate shifts, while the cascade found the one that maximises the
whole-patch ZNCC, the score the member was kept on. Applied unconditionally, the
cell fit's map moved 80% to 99.6% of kept members on the fleet and lowered every
moved member's ZNCC. With the acceptance rule below it moves 0.2% to 1.4% of
kept members, each to a higher ZNCC, and still changes the seed: on
`KerryPark480` the 55 members it moves, 0.34% of the kept members, alone turn
the seed's passing pick into a failing one. ZNCC cannot say whether those shapes
are worse, so by default the stage measures and leaves every shape where the
cascade put it.

#### The shape-moving loop (`move_shape`)

With `move_shape`, the fitted map is an update of the member's shape and the
stage runs on two levels: the outer level renders, the inner level reads only
the working patch.

1. **Render** at the current shape, and **register the cells**, as in the
   measurement.
2. **Fit the update.** The robust affine map through the surviving cells, in
   grid coordinates centred on the grid's middle (`c ↦ A·c + b`), composes into
   the shape as `S' = S·A` and the position as `p' = p + S·(step·b)`.
3. **Accept or reject.** The whole-member windowed ZNCC, the objective the
   cascade maximised, is read at the updated shape. If it is lower than at the
   current shape by more than `ACCEPT_ZNCC_TOLERANCE`, or lower than at the
   starting shape at all, the update is rejected, the shape is not moved and the
   loop stops (`Rejected`). The tolerance lets the loop cross a flat stretch
   above the start; the floor at the start keeps it from accumulating over
   several updates. A shape whose support leaves the frame cannot be read and
   counts as the lowest score.
4. **Loop.** The loop stops with the update applied if it moves no cell centre by
   more than `update_tolerance_px` (`Converged`), or if it was the last of
   `max_iterations` renders (`Cap`). It stops if the update's largest
   cell-centre movement is not smaller than the previous update's
   (`Oscillation`): the loop is alternating between shapes rather than
   converging, and it keeps whichever of the shapes before and after the update
   has the higher whole-member ZNCC, the shape after on a tie. Otherwise the
   shape is updated and the loop renders again.

This is the inverse-compositional alignment pattern with a piecewise
translation model. On a synthetic plane, from a starting shape a few percent of
scale and a few degrees off, it converges in two or three iterations. On
captures, starting from the cascade's shape, it almost always stops at its first
iteration, because its first update lowers the score the cascade maximised.

**The stored displacement after the loop.** It keeps its definition, the
displacement from where the returned shape places the cell. When the last update
was applied, a shift `d` measured at cell centre `c` is carried into the
returned shape's grid as `A⁻¹·(d − (A·c + b − c))`, so the update's affine part
is gone from it because the shape absorbed it; subtracting the update's movement
without the `A⁻¹` would leave an error of the size of `(A⁻¹ − I)` applied to the
displacement. When the last update was rejected, or an oscillation kept the shape
before it, the returned shape is the one the last render was made at, and the
shift is stored as measured.

**Revert.** After the loop moves a member, its whole-patch ZNCC, its parts and
its shift from the seed are read again at the new shape, and the cascade's
gates, `min_zncc` and `max_shift_px`, are applied to them. A member that fails
either, whose new support leaves the frame, or whose re-read ZNCC is below the
cascade's stored ZNCC (the loop's floor is its own reading of the starting
shape, which can differ from the stored value in the last bits) keeps its
cascade shape and readings, with all nine cells `not_attempted`. The same
happens when any iteration fails: a failed render, no surviving cell, or a
fitted map that reflects (`det A ≤ 0`) or is not finite. What earlier iterations
fitted is discarded, because it was read at a shape that would not be the one
returned. A rejected update is not a failure: the member keeps the shape the last
render was made at, with that render's readings. The member's status is always
the cascade's.

#### What the fleet measurements decide

The measurements are in
[cluster-patch-refinement-measurements.md](cluster-patch-refinement-measurements.md).

- **The loop without an acceptance rule**
  ([fleet conclusions](cluster-patch-refinement-measurements.md#what-these-measurements-decide),
  43 entries): it moved 80% to 99.6% of kept members, by a median of 0.16 to
  0.71 grid px and up to 10, lowered every moved member's whole-member ZNCC, left
  23.9% of members alternating at the cap, cost 1.6× the refinement's time, and
  turned the seed's `KerryPark480` pick from pass to fail.
- **The loop with the acceptance rule, the robust fit and the oscillation stop**
  ([subset run](cluster-patch-refinement-measurements.md#subset-with-the-acceptance-rule-2026-10-08),
  five entries): 98.5% to 99.8% of kept members stop at their first iteration,
  under 1.5% move, each to a higher whole-member ZNCC, nothing reaches the cap
  or oscillates, and the cost is 1.12× to 1.29×. `KerryPark480`'s pick still
  fails, and moving only the 55 members the loop moved there reproduces the
  failure.
- **The measurement, the default**
  ([subset run](cluster-patch-refinement-measurements.md#subset-with-the-shape-left-to-the-cascade-2026-10-08)):
  every member output other than the cells is byte-for-byte the cascade-only
  run's on all five entries, the CPU cost is 1.09× to 1.17×, and the seed on
  `SeoulBull` and `KerryPark480` releases the same candidates as with the
  cascade file, so `KerryPark480`'s pick passes again.
- **The two-dimensional residual scale**
  ([subset run](cluster-patch-refinement-measurements.md#subset-with-the-two-dimensional-residual-scale-2026-10-08)):
  the robust fit refuses 2.6% to 3.4% of cells as outliers instead of 0.5% to
  0.7% under the one-dimensional factor, and changes no other cell status, shift
  or ZNCC.
- **The cell gates.** Neither the cell ZNCC distribution nor the curvature's has
  a valley or a knee to place a bar in
  ([gate sweep](cluster-patch-refinement-measurements.md#gate-sweep)); `0.8`
  refuses 6.8% of peaked cells, and `0.02` refuses 2% to 14% of cells depending
  on the capture.

## Implementation notes

**Determinism is a contract, not an accident.** Clusters refine in parallel with
rayon over disjoint member ranges, per-cluster scratch lives inside the closure,
and every tie is broken by index: reference selection sorts by scale then global
member index, the per-image dedupe compares with a strict `>` so the earlier
member wins, the simplex reorder is a stable insertion sort, and the returned
optimum is the first minimum (the numpy-argmin convention). Two runs over the
same input are bit-identical under any thread schedule.

**The status discriminants are a cross-crate invariant.** `MemberStatus` is
its own enum, not `sfmtool_matches_format::ClusterMemberStatus`; the binding
casts it to `u8` and writes it straight into the `cluster_patches/` section. The
two enums must stay numerically identical, which a `const` assertion in
`params.rs` checks at compile time, and a new status has to land in both, plus
the format's validator, in one change. `CellStatus` and
`ClusterCellStatus` share their discriminants the same way: `member_cell_data`
converts one to the other, and a test checks that the codes agree.

**The tile bound is the frame test.** Each evaluation samples through a
per-(member, level) tile: a planar f32 copy of the touched region of that pyramid
level, built once when the level is first reached and grown lazily to cover each
evaluation's bounding box. Because the tile is always a subset of the image and
always covers the evaluation's *clipped* footprint, an out-of-tile tap is exactly
an out-of-frame tap — the lane-bounds test doubles as the all-in-frame test and
as the early-out, and the hot loop needs no separate image-bounds check.

**The tile is centered for a numerical reason.** Windowed ZNCC needs a variance,
accumulated as `S2 − S1²/W` in f32. On raw 0–255 intensities that difference
cancels catastrophically for low-contrast patches; subtracting the tile mean at
conversion time keeps the accumulation well-scaled, and the windowed ZNCC is
shift-invariant so nothing has to be undone afterwards.

**Scalar and AVX2 must agree, and the agreement is structural.** The scalar path
is the reference implementation, the non-x86 fallback and the dual-path test's
oracle; the runtime-dispatched AVX2+FMA path is a restructuring, not a
re-derivation. It fuses all template channels into one k-major pass so the
channel-invariant work — coordinate FMAs, the in-frame mask, tap indices, blend
weights — is computed once per 8 support points, while keeping each channel's
accumulation order identical to the channel-major original, which is what makes
the fusion bit-exact rather than merely close. The four bilinear taps come from
64-bit pair loads, since the two horizontal taps of a lane are adjacent floats:
half the fetched elements of a 32-bit hardware gather, and plain loads sidestep
the microcoded gather penalty on hybrid parts, where most rayon threads land.
Non-finite coordinates from a degenerate warp convert to `i32::MIN` and are
caught by the same lane mask; the scalar path checks them explicitly.

**The sampler is local on purpose.** The kernel reuses the house pixel-center
bilinear convention, the shared window support and the `weighted_moments_pub` /
`znorm_write` z-normalization — but not `view_selection`'s
`sample_support_affine`, whose contract (border-gated maps, no validity
reporting, `u8` re-rounding for remap parity) is incompatible with a sampler that
must report out-of-frame and keep continuous values. Nor is
`score_raw_against_reference` shared: the fused sample+reduce loop realizes the
same algebra in one pass, and splitting it would reintroduce the intermediate
buffer the fusion exists to remove.

**The optimizer allocates nothing.** The simplex lives in fixed `[f64; 6]`
buffers (the affine stage is the widest) and the per-iteration reorder is an
in-place stable insertion sort. At the order of 10⁸ objective evaluations in a
full run, the per-iteration `Vec` churn of the original transcription cost about
as much as the arithmetic; removing it changed no result.

**Profiling is opt-in and free when off.** `cluster_refine::prof` carries the
house phase timers and counters — gate, template, cascade, tile builds and their
pixel volume, evaluations per cascade stage — behind `SFMTOOL_PROFILE=1`,
compiling to one branch on a cached flag when the variable is unset.

### Measured cost

On dino_dog_toy (85 images at 2040×1536, 105,326 clusters, 373,194 members,
i9-14900HX, 32 threads) the kernel runs in 3.2 s wall / 103 CPU-s, and the whole
`sfm cluster-patches` invocation in 7.8 s. Three changes took it there from
6.1 s / 194 CPU-s: the fused pair-load AVX2 kernel (1.93 → 1.15 µs per objective
evaluation, bit-identical), the allocation-free Nelder-Mead (−30 CPU-s,
bit-identical), and the cascade stopping rules — the only one of the three that
changes results. Their sweep: evaluations 265 → 213 per member, kept members
+0.03%, mean kept ZNCC −0.0001, warp-consistency median 0.0677 → 0.0669 and p90
0.1993 → 0.1972 (slightly better), status flips 0.76%. A tighter
`stall_tol = 2e-4` was rejected because it broke the synthetic scale-1.25 /
rotation-20° recovery case.

Roughly 60% of kernel CPU is still the objective, most of it the affine stage's
cap-bound crawl. Two candidates were considered and not pursued: replacing that
stage with a Gauss-Newton/ECC step on an analytic windowed-ZNCC gradient, and
luminance-only refinement (3× fewer channel passes, but it changes matching
semantics and needs its own quality study).

### The member gate's effect and cost

Measured on the clusters the track-at-pixel harness builds from each ground
truth's own index (`scripts/track_at_pixel/dataset.py`: seoul_bull's 17 images,
and kerry_park's 48 fisheye frames of candidate `tk113`), with every other
setting at its default.

| gate | dataset | refused | refinable clusters | clusters keeping a member | reference + kept | gate CPU per member |
|---|---|---|---|---|---|---|
| off | seoul_bull | 0 | 4,942 | 2,183 | 8,169 | 0 |
| radius ≤ 2.5 | seoul_bull | 1,201 (9.6%) | 4,500 | 1,823 | 7,096 | 53 µs |
| off | kerry_park | 0 | 14,288 | 9,294 | 31,091 | 0 |
| radius ≤ 2.5 | kerry_park | 5,335 (12.8%) | 12,434 | 7,929 | 26,543 | 52 µs |

The radius at `2.5` refuses about one member in nine. Reading it costs about
52 µs of CPU per member, summed over threads, split about evenly between
sampling the `R × R` grid and reading it, but the members it refuses are never
refined, so the kernel as a whole spends less CPU with the gate on than off
(seoul_bull 5.6 → 5.3 CPU-s, kerry_park 20.3 → 19.6; 0.17 and 0.68 s wall,
i9-14900HX, 32 threads). Measured with the gate reading a tile with a ring of
`r` px around the member grid, the resections the add-image-to-tracks harness
runs over these files move little: at the resected pose its default rule recovers
80.4% → 80.6% of the known observations on seoul_bull and 71.5% → 70.7% on
kerry_park, and the recovery at the ground-truth pose is unchanged.


### Piecewise stage internals

- **Determinism and precision.** Tile intensities are `f32`, matching the
  cascade; the affine fit, a weighted least squares on at most nine points run
  four times for the reweighting, is `f64`, and with `move_shape` it is composed
  into the `f32` shape, where a near-identity matrix is multiplied repeatedly.
  Per-member work is independent and parallel, with no randomness; a test checks
  that one thread and four give the same cells. Without `move_shape`, every
  member output other than the cells is bit-identical to a run without the
  stage, which a kernel test checks.
- **Without `move_shape` the stage never reads the whole-member ZNCC.** The loop
  returns after its one render and fit, before the score is read, and the
  caller sees an unmoved member and stores only its cells. That is what keeps
  the cascade's outputs bit-identical, and why the stage costs one render and
  nine searches per kept member and nothing more.
- **The render is the only cache-missing step.** The working patch is rendered
  once per member without `move_shape` and once per iteration with it, and
  nothing in the cell search reads the photograph. A cell's search reads the
  window's sum and sum of squares from summed-area tables of the working patch,
  built once per render; the template side is mean-removed, so the cross term
  needs no window mean and is the only pass over the window per shift.
- **The update's acceptance reads the cascade's own evaluation**, the same
  objective at the map the shape and position give, through the same
  pyramid-level choice and tile; the loop reads it once at the starting shape
  and once per iteration. A first version applied the tolerance per update, and
  on the subset the stored ZNCC of members moved over several updates fell by up
  to 0.0004; the floor at the starting shape is why it no longer can.
- **The loop's displacements follow the second-order term less closely as it
  grows.** On the synthetic plane with `h = [0.003, −0.002]` the loop converges
  with its last update applied, and the largest error over the nine cells is
  0.093 grid px (RMS 0.058) against true displacements up to 0.64: the robust
  fit weights the corner cells, where the term is largest, below the others, so
  the affine it leaves follows the middle cells. At twice that the second update
  is rejected, and the corner cell farthest off the affine is refused as an
  outlier.

## Parameters

Defaults are `ClusterRefineParams::default()` in
`crates/sfmtool-core/src/patch/cluster_refine/params.rs`, except for the two
module constants the last column marks. The CLI's `--patch-size` is the **full**
template edge length while the kernel's `radius` is a half-width;
`src/sfmtool/_cluster_patches.py` is the sole conversion site
(`radius = patch_size / 2`).

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `radius` | `6.0` | Template half-width, keypoint-frame units (CLI `--patch-size 12.0`) |
| `resolution` | `25` | Support samples per axis; the kernel clamps up to 2, the CLI to ≥ 3 |
| `window` | `GaussianDisk { sigma: 0.5 }` | Scoring window; sigma in normalized patch coordinates, where the grid spans `[−1, 1]²` |
| `min_zncc` | `0.85` | Acceptance threshold on the achieved windowed ZNCC |
| `max_shift_px` | `3.0` | Max translation drift from the SIFT seed, source px |
| `max_member_zncc_self_similarity_radius` | `2.5` | Member gate on the ZNCC self-similarity radius of the member's own patch, template-grid px; `0` disables it (CLI `--max-member-zncc-self-similarity-radius`) |
| `max_iters` | `120` | Nelder-Mead iterations per cascade stage |
| `convergence` | `1e-5` | Simplex value-spread stop, affine stage |
| `intermediate_convergence` | `1e-4` | …and for the shift and similarity stages, which only seed the next |
| `stall_iters` | `20` | Iterations without progress before a stage exits |
| `stall_tol` | `1e-4` | Best-value improvement (ZNCC units) that counts as progress |
| `MIN_ABS_DET` | `1e-9` | Floor on a usable SIFT shape's `det A` magnitude (`cluster_refine.rs`) |
| `SIGMA_CLAMP` | `1.5` | Log-scale clamp of the similarity stage (`cluster_refine.rs`) |

The piecewise stage is `ClusterRefineParams::piecewise`, `None` by default (CLI
`--no-piecewise`). Its settings are `PiecewiseParams::default()` in
`crates/sfmtool-core/src/patch/cluster_refine/piecewise.rs`, and the module
constants the table marks are in the same file. The defaults of
`min_cell_zncc` and `min_cell_curvature` are `DEFAULT_MIN_CELL_ZNCC` and
`DEFAULT_MIN_CELL_CURVATURE`. Both are set on the synthetic tests; the fleet's
distributions of cell ZNCC and curvature have no valley or knee that would place
them ([gate sweep](cluster-patch-refinement-measurements.md#gate-sweep)).
`sfm cluster-patches --piecewise` runs the
stage at these defaults and exposes none of them.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `move_shape` | `false` | Whether the fitted affine map may move the member's shape, by the loop of [The shape-moving loop](#the-shape-moving-loop-move_shape); off, the stage measures once at the cascade's shape |
| `cell_shift_bound_px` | `2.0` | Search bound for a cell's shift from its affine placement, and the working patch's margin, template grid px |
| `min_cell_zncc` | `0.8` | A cell below this ZNCC at its optimum is `refused_zncc` |
| `min_cell_curvature` | `0.02` | A cell whose ZNCC peak is flatter, in ZNCC per grid px², is `refused_curvature` |
| `update_tolerance_px` | `0.05` | With `move_shape`: the loop stops when no cell centre moves by more, grid px |
| `max_iterations` | `5` | With `move_shape`: render cap; without it the stage renders once |
| `ACCEPT_ZNCC_TOLERANCE` | `1e-4` | With `move_shape`: how far the whole-member ZNCC may fall at an update (module constant) |
| `IRLS_ROUNDS` | `3` | Reweighting rounds after the curvature-weighted first fit (module constant) |
| `TUKEY_CUTOFF` | `4.685` | Biweight cut-off, residual scales (module constant) |
| `MEDIAN_LENGTH_TO_SIGMA` | `0.8493` | Median residual length to per-axis scale, `1/√(2 ln 2)` (module constant) |
| `MIN_RESIDUAL_SCALE_PX` | `0.1` | Floor on the residual scale, grid px (module constant) |

## Python bindings

`_sfmtool.matching.refine_cluster_patches`, in
`crates/sfmtool-py/src/matching/cluster.rs`, registered beside
`background_floor_clusters` and `clusters_to_pair_matches`:

```python
refine_cluster_patches(
    images, positions, affine_shapes,
    cluster_starts, member_images, member_features, *,
    radius=6.0, resolution=25,
    window="gaussian_disk", window_sigma=None,
    min_zncc=0.85, max_shift_px=3.0,
    max_member_zncc_self_similarity_radius=2.5,
    max_iters=120, piecewise=False, move_shape=None,
    cell_shift_bound_px=None, min_cell_zncc=None, min_cell_curvature=None,
    update_tolerance_px=None, max_iterations=None, progress=None,
) -> dict
```

`images` is one `HxW` or `HxWxC` uint8 array per image, in the images-section
order the cluster arrays index; `positions` and `affine_shapes` are parallel
lists of `(N, 2)` and `(N, 2, 2)` float32 arrays; the three cluster arrays are
uint32. `window` is `"gaussian_disk"` (default), `"gaussian"` or `"uniform"`, and
`window_sigma=None` resolves to `0.5`. The four cascade-tuning knobs
(`convergence`, `intermediate_convergence`, `stall_iters`, `stall_tol`) are
deliberately not exposed: they trade evaluations against a kept set that has been
swept once, so moving them changes results rather than only cost.

Argument names match the Rust ones. Parallel-list lengths, per-image array
shapes, CSR self-consistency and the `member_images` range raise `ValueError`
before the GIL is released; out-of-range `member_features` and degenerate shapes
are data, not errors, and reach the caller as `not_evaluated`. Pyramids are built
through `patches::views::build_pyramids_from_image_list`, and the kernel runs
under `py.detach`.

The returned dict is member-parallel: `reference_members` `(C,)` uint32
(`0xFFFFFFFF` = unrefinable), `member_status` `(M,)` uint8,
`member_positions` `(M, 2)` float64, `member_affine_shapes` `(M, 2, 2)`
float64, `member_zncc` `(M,)` float32, `member_zncc_middle` `(M,)` float32,
`member_zncc_grid` `(M, 3, 3)` float32,
`member_shift_px` `(M,)` float32, and
`member_consistency_residual` `(M,)` float32 — the warp-consistency signal,
computed inside the same call.

`piecewise=True` runs the piecewise stage; the six `PiecewiseParams` settings
are keyword arguments under their field names, each `None` for the Rust default,
and are ignored without `piecewise`. With it the dict also carries the four
stored columns, `member_cell_shift_px` `(M, 3, 3, 2)` float32,
`member_cell_zncc` `(M, 3, 3)` float32, `member_cell_status` `(M, 3, 3)` uint8
and `member_cell_iterations` `(M,)` uint8; `piecewise_options`, a dict of the
settings it ran with; and two readings the file does not store,
`member_cell_loop_stop` `(M,)` uint8 (the `LoopStop` code, `5` for every kept
member without `move_shape`) and `member_cell_update_accepted` `(M,)` bool.
Without it those seven keys are `None`.

```python
from sfmtool.matching import refine_cluster_patches

out = refine_cluster_patches(
    images, positions, affine_shapes,
    cluster_starts, member_images, member_features,
    radius=6.0, min_zncc=0.85,
)
kept = out["member_status"] == 1                    # 1 == kept
shapes = out["member_affine_shapes"][kept]          # absolute 2x2 shapes
points = out["member_positions"][kept]              # refined positions
```

## Testing

`cluster_refine/tests.rs` covers the kernel: **synthetic recovery** across the
calibrated warp range (scale 0.8–1.5×, rotation ≤ 20°, shear ≤ 0.15) with the
seed perturbed by the experiment-observed noise (`|Δlog s|` 0.07, `|Δrot|` 4°,
1 px shift), recovering `W = S·S_ref⁻¹` through the reference member exactly as a
consumer would and asserting a support-grid RMSE around 0.3 px; **one test per
gate** (flat member image → `RejectedLowZncc`; seed drifted past `max_shift_px`
→ `RejectedShift`; support out of frame → `NotEvaluated`; an unlocalizable
member excluded, and unable to become the reference; a border member still
read through the clamped sampling; every reference candidate out of frame →
unrefinable; a degenerate cluster → not evaluated; two members in one image →
exactly one `Kept`); the **member gate** (a textured patch reads well under the
bar while a flat patch, a straight edge and a texture smooth on the template
grid read over it; an edge surrounded past the grid by the same edge inverted reads as the edge
alone, while a reading with a ring around the grid reads it under 3; at the default a flat
and an edge member are refused and a textured one kept, `0` refuses nobody and
`3` turns nothing out; the pass rule at, over and under the bar and for `NaN`);
**determinism**, two runs bit-identical; and a **dual-path**
check that AVX2 and scalar scores agree within 1e-4. The synthetic `texture`
carries two fine terms (periods near 6 px) beside its smooth ones, because the
smooth terms alone read over `2.5` on the template grid and the default gate
would refuse every member; `smooth_texture` keeps the smooth terms alone for
the test that shows it. That the low-ZNCC test uses
a *flat* member image rather than an unrelated smooth texture is behaviour, not
test convenience: over the ~50 effective samples of the window the affine
optimizer can chase an unrelated smooth texture to a spurious ZNCC above the
permissive 0.85 gate, which then trips the shift gate instead.

`cluster_refine/piecewise/tests.rs` covers the piecewise stage. **The
measurement**, on a synthetic planar cluster seen from a tilted view with the
starting shape perturbed by a known affine: one render, no whole-member ZNCC
read, the shape and position left at the start, all nine cells fitted, and the
stored displacements matching the perturbation within a fifth of a grid px.
Inside `refine_cluster_patches`, every member output but the cells is
bit-identical with the stage on and off, and the kept member reads one
iteration and `Measured`. A cell off the affine the other eight agree on is
`refused_outlier` with its shift stored. **The loop** recovers the perturbed
affine in at most three iterations with all nine cells fitted, and its cells
follow the homography's second-order term; with a third of the template over a
second plane, the three cells over it are refused and the update follows the
first plane. An update that lowers the whole-member ZNCC, or whose support
leaves the frame, is rejected; alternating updates stop at the second iteration
with the better shape; a refined shape that fails a cascade gate is reverted;
a flat member keeps its cascade shape with nine cells not attempted. Further
tests pin the model fallbacks (similarity for three or four survivors, for five
in one row, and for a singular affine; a shift for one or two), the
summed-area window sums against a direct pass, the Rayleigh scale factor, the
shared cell centres, the shared status codes, `member_cell_data`, and equal
cells on one thread and four.

`consistency/tests.rs` covers the residual fit (oracle cameras fit exactly,
absolute shapes reproduce the relative-warp residuals, a contaminated member
scores highest, non-participants NaN, runs deterministic).
`tests/rust_bindings/matching/test_cluster_patches_rust_bindings.py` pins the dict schema,
dtypes, progress ticks and every `ValueError` path, and with `piecewise` the
cell keys, the `move_shape` loop and the settings' pass-through.
`tests/patch/test_cluster_patches.py` drives the command over the
`isolated_seoul_bull_17_images` fixture through the real pipeline
(`ws init` → `sift --extract` → `match --cluster` → `cluster-patches`) and
asserts that the output verifies, that over half the multi-member clusters keep
at least one member, and that statuses stay inside the enum; with
`--piecewise` it checks that the cells are written.

## Non-goals

- **No matching.** The kernel refines the clusters it is given; producing them is
  [track-cluster matching](../features/track-cluster-matching.md)'s job, and a
  refinement-knob change must never force a re-match — the reason the two are
  staged artifacts.
- **No perspective warp, and no multi-view congealing pass.** Both were measured
  against the calibration data, and both are dead ends at this scale.
- **No reconstruction.** The operation runs before any pose exists, so the
  geometric consistency it can offer is the reconstruction-free residual, not a
  reprojection test.
- **No gate on consistency.** The residual is stored, never thresholded here.
- **No piecewise refinement by default.** The stage runs only when
  `ClusterRefineParams::piecewise` is set (`sfm cluster-patches --piecewise`),
  and by default it measures without moving any shape (`move_shape` off).
- **No consumer reads the cells.** No pipeline stage, the seed's writer
  included, derives frames or normals from them or calls
  [cell-plane-normals.md](cell-plane-normals.md)'s kernel; frames built from
  the cells behind a precision gate are proposed in
  [cell-plane-normal-precision-gate.md](../../drafts/cell-plane-normal-precision-gate.md).
- **No normal in the cluster-patches file.** The file is pose-free; a normal is
  derived by whoever holds poses.
- **No affine fit per cell.** A cell's search is a shift search; a cell that
  needs more is over another surface and is refused.

## Open questions

- **The member gate's unit is not resolution-invariant.**
  `max_member_zncc_self_similarity_radius` is in template-grid px, and the
  shifts the reading searches are whole grid px, so the same bar asks for a
  different image-space sharpness as the template is sampled more or less
  finely. The member gate the radius replaced, whose bar was in grid px too,
  showed the size of the effect: on dino_dog_toy, moving from 15 to 31 samples
  per axis at a fixed bar cut `RejectedUnlocalizable` from 1,913 members to
  372. The radius has not been measured this way. Since `--resolution` is
  freely tunable, one knob moves another gate's strength. Re-expressing the bar
  in a resolution-independent unit (keypoint-frame or source px) would fix it,
  and would change the meaning of the current default.
- **Shift only, or shift and scale, per cell.** A scale per cell would read the
  perspective term's radial component, at the cost of a two-dimensional search
  per cell. The cells are fitted as pure shifts.
- **Reference-selection policy.** Largest SIFT scale is the shipped policy and a
  known weakness on rig captures, where the largest-scale member is often an
  untracked feature. The format is policy-agnostic; the alternatives (template
  self-agreement, descriptor centrality) are the design spec's question to
  settle.
