# ZNCC Self-Similarity Radius

A keypoint is only worth keeping if matching can find it again: when the patch around it is compared with another photograph, the best match should sit at one place and nothing else nearby. The ZNCC self-similarity radius measures that from a single image. It slides the patch over itself by whole pixels, up to a small maximum distance, and asks how far it can move while still looking as much like itself as a true match between two views would. A corner or a busy texture stops matching itself within a pixel and scores 0 or 1. A straight edge keeps matching itself along its own length and scores the maximum, and so does a flat sky, whose only texture is noise and rounding. Alongside the distance, the score reports the direction the patch can slide in, and the patch's ZNCC with itself at every shift it tried. It is computed for the whole patch, its middle, and each ninth of it, like the bench's other patch readings.

It is the score sfmtool judges a keypoint's position by. The classical score, the curvature of the same surface at its peak, reads a strong straight edge as sharp, and is not used ([Why a curvature is not enough](#why-a-curvature-is-not-enough)). The bench judges the radius, with its `max_zncc_self_similarity_radius` bar. The keypoint localizer's member gate, `max_member_zncc_self_similarity_radius`, Add Image to Tracks, which uses the same gate, and cluster-patch refinement's member gate, `ClusterRefineParams::max_member_zncc_self_similarity_radius`, also judge the radius, with a default bar of 2.5 patch-grid px, the bench's default bar too. So do the two culls on a point's stored consensus bitmap, `embed-patches --max-zncc-self-similarity-radius` and `xform --filter-by-zncc-self-similarity-radius`. Every one of them reads exactly the `R×R` bitmap it judges, with no pixels from outside it, by [the overlap reading](#the-overlap-reading): the bench the tile it renders for an observation, the member gates the view's or the member's own grid, and the culls the stored bitmap. So the radius is the self-similarity of the bitmap itself, and the same number whichever of them computes it.

## What it measures

### The overlap reading

Take a tile with channels `I_c`, a flag on each sample saying whether it carries data, and a template rectangle `Ω` inside the tile. The tile is the bitmap being judged: the tile the bench renders for an observation, the view's or member's own grid a member gate reads, or a point's stored consensus bitmap (the `.sfmr` `patch_bitmaps_y_x_rgba` row, or the round-1 consensus `embed-patches` fuses), each exactly `R × R`. The template is the whole bitmap or one of its parts (below). Let `r` be the maximum radius. For every whole-pixel shift `d = (dx, dy)` of the square `|dx|, |dy| ≤ r`, the **overlap** `O_d` is the set of samples `k` of `Ω` where `k` and `k + d` both lie inside the tile and both carry data. The template is compared with the window moved by `d` over that overlap, one channel at a time:

```
z_c(d) = Σ_k (t_c(k) − a_c,d) · (u_c(k) − b_c,d)  /  sqrt( Σ_k (t_c(k) − a_c,d)² · Σ_k (u_c(k) − b_c,d)² )
z(d)   = mean of z_c(d) over the template's textured channels
```

The sums run over `k` in `O_d`, `t_c(k) = I_c(k)` is the template and `u_c(k) = I_c(k + d)` the moved window in channel `c`, and `a_c,d` and `b_c,d` are their means over `O_d`, so every `z(d)` is a ZNCC on one scale and `z(0) = 1`. This is the ZNCC the pipeline matches with, per channel and averaged, with every sample weighted equally. Nothing outside the tile is read: past a bitmap's edge there is either nothing rendered or the patch's surroundings, and a radius that read them would describe more than the bitmap.

A template with `r` px of tile around it, every sample carrying data, has `O_d = Ω` at every shift: the template stays whole and the moved window reads the tile around it. A bitmap's middle square (a margin of `R/4 ≥ r` at `R ≥ 12`) and its centre cell (`R/3 ≥ r`) are such templates.

- **Data.** A sample carries data when its alpha (the fourth channel) is above 0; a tile with no fourth channel has data at every sample, as a rendered tile does. The consensus fuse writes alpha 0 where no view covered the sample, where only one view did, and where the views disagree so much that the confidence rounds to 0, and its colour there is zero or one view's reading, so all of those drop out, as a sample past the edge does. A row for a point with no consensus is zero throughout, alpha included. Samples with alpha 0 are rare in consensus bitmaps: 0% on seoul_bull and 0.16% on kerry_park.
- **Channels.** The colour channels are read, up to three; alpha is read only as the data flag. Colour is what matching sees, so a red-and-green edge of equal brightness, which is flat in luminance, is scored as the edge it is. The channels judged are those whose spread `s_c` over the template's samples with data (`O_0`) is at or above the flat floor. At each other shift a channel whose template spread over `O_d` falls under the floor is left out of that shift's average as well, and a moved window with no variance in a channel (a centred sum of squares under `1e-6`, the matching ZNCC's flat constant, or under the summed-area tables' rounding where the [dense route](#implementation-notes) takes it from them) scores 0 in it, since a flat window is plainly different from a textured template. A shift whose overlap leaves no channel, or has no sample at all, scores 0: nothing there matches the template.
- **The parts.** The whole bitmap is read as one template, its middle square and the nine cells of the ZNCC grid's split as others, each taking its moved windows from the rest of the bitmap where it reaches: at each shift a part's sample counts when its moved sample lies inside the bitmap and both carry data. The whole bitmap and an edge or corner cell lose the rows and columns their moved window would read past the bitmap. The whole bitmap's overlaps keep at least `(R − r)/R` of it on each axis; an edge or corner cell of side `c` (about `R/3`) keeps at least `(c − r)/c` of itself on each axis it loses rows or columns on, 5/8 at `R = 24`, `r = 3`.
- **No data.** A template with no sample carrying data has **no reading**: its radius, slide, tolerance and surface are all `NaN`.

### The radius, the slide and the surface

A shift is **indistinguishable** from the true position when its ZNCC deficit is within the tolerance:

```
1 − z(d) ≤ τ ,     τ = ε + mean_c (n / s_c)²
```

where `s_c` is the template's own standard deviation in channel `c` in grey levels over `O_0`, the mean runs over the same textured channels as `z`, `n` is the noise between two views in grey levels, and `ε` is the relative mismatch between two views of the same surface (see [Theory](#theory)). `ε` and `n` are parameters of the score, passed in `SelfSimilarityParams` and the same for every patch in a call, so the radius is a property of the patch alone: it depends on the tile and the parameters, and on nothing about the views or the track the patch belongs to. `τ` uses the spread of the whole template, not the spread over each shift's overlap: it is the deficit a true match of this template between two views would show, which is a property of the template at its true position, where the overlap is the whole template. It also has to be one level for the whole surface, since the radius is read where the surface, interpolated between neighbouring shifts, falls through `1 − τ`, and a level that changed from shift to shift would have no single crossing to interpolate. How far an overlap's spread differs from the template's depends on the texture, and it can differ most where the overlap keeps least of the template, as at an edge or corner cell.

The **radius** is the largest distance from the centre of the contour where the surface falls through the level `1 − τ`, the isosurface of the indistinguishable shifts, as well as the whole-pixel readings can place it. Over every grid edge of the square between two neighbouring shifts, one at or above the level and the other below it, the ZNCC is interpolated linearly along the edge to the point where it equals the level; the radius is the largest distance of those points from the centre, capped at `r`. The centre is always at or above the level, so a patch that locks reads the fraction of a pixel its peak takes to fall through it rather than 0. A shift at or above the level whose neighbour was not read has its crossing in that direction somewhere past itself, so its own distance counts as a lower bound for the crossing. A neighbour is not read when it lies past the square's border, and every border shift is at least `r` from the centre, so a region at the level that runs off the square reads `r`, read as "`r` or more": the patch still matched itself at the edge of the search and may slide further. A neighbour whose reading is not finite is treated the same way; it is not a crossing, since there is no value to interpolate to. The overlap reading never produces such a reading, even on a tile carrying non-finite pixels: a non-finite sample that carries data makes its channel's tile mean non-finite, so that channel falls under the flat floor and drops out of the reading, and a tile with one in every colour channel reads as having no texture. A surface handed to the contour from elsewhere can hold one. A template with no channel at or above the flat floor at `d = 0` has no texture to match and scores `r`, with a slide of `[0, 0]`, an infinite tolerance and a surface of `NaN`.

The **slide** is the direction of the indistinguishable shifts. With `C = Σ d dᵀ / |A|` over the set `A` of indistinguishable shifts of the square, the same shifts the radius reads, and eigenvalues `μ₁ ≥ μ₂`, the slide is the unit eigenvector of `μ₁` scaled by `1 − μ₂ / μ₁`, in the grid frame (`x` column-right, `y` row-down). A set strung out along a line gives a slide near unit length along it, a set spread evenly around the centre gives one near zero, and an empty set gives `[0, 0]`. Its sign means nothing.

The **surface** is `z(d)` itself over the `(2r + 1)²` square of shifts, corners included, row-major from `(dx, dy) = (−r, −r)`, with `1` at the centre. The radius and the slide are read from all of it, and a display draws the same square, so the contour it draws at `1 − τ` is the one the radius is read from.

## The contour and its reach

The radius is one length in grid px. The same contour also bounds where a match could land along each axis, and in the units of the photograph and of the scene, which is what a confidence bound on the 3D point along the patch's axes needs. The reading's **contour** is the set of points the radius is read from:

- every crossing interpolated along a grid edge, as above, at its offset `(dx, dy)` in the grid's axes (`x` column-right, `y` row-down);
- every shift at or above the level whose neighbour one grid step `e` away has no reading, at the shift's own offset, marked **open** towards `e` (`ContourPoint::open_towards`): its crossing lies somewhere past it in that direction, so its offset is a lower bound for it along that axis;
- for a template with no textured channel, whose every shift counts, every shift of the square's border, open towards the outside.

The radius is the largest distance of a contour point from the centre, capped at `r`, which is the radius above to the bit. A reading with no data (the overlap reading's `NaN`) has no contour. The **reach** of the contour is measured from it in three units, each length a `BoundedLength`, a value and whether it is only a lower bound (`at_least`):

- **Grid px.** The radius, and the reach along each grid axis, `max |dx|` and `max |dy|` over the points.
- **Image px.** With `J` the Jacobian of the map from the tile's grid to the source image at the tile's centre, in image px per grid px, the image radius is the largest `|J · d|` over the points `d`, each first brought in to `r` from the centre where it lies further, the cap the radius takes, so that under `J = s·I` it is `s` times the radius. A Jacobian that is not finite or is singular gives no image radius. Because `J` applies to the whole contour rather than to its furthest point, a contour stretched along `x` changes its image radius under a stretch of the image along `x` and not under one along `y`.
- **Along the patch's axes.** For an `R × R` bitmap rendered through the patch `placement`, one grid px along `x` is `2·half_extent[0]/R` along `u`, and one along `y` is `2·half_extent[1]/R` along `−v` (`WarpMap::from_patch` steps the rows down `v`; the sign does not matter for a reach). The grid reaches scale to the reach along `u` and `v`, with their lower bounds, as lengths in the scene's world-space unit for a finite patch (`world_space_unit`, or scene units where the file names none). A patch at infinity (`w = 0`) has no length: its `center` is a unit direction `d`, its axes are tangent offsets, and the corner `d + a·u` is the direction at `atan(a)` from `d`, so its reach reads `atan(reach)` as an angle in degrees, which is the offset in radians to first order.

### Lower bounds

The reach is meant to bound where the 3D point could be along the patch's axes. A value is exact within the square of shifts searched, with a region that runs off the square and holds its width (below) taken to keep that width past the border; wherever the readings leave room for a larger value, it is flagged `at_least`. A repeating texture whose period exceeds `r` can match itself again beyond the square, and that is not seen: no reading of the square can tell it from a patch that does not repeat. Two things can hide part of the contour: the region at the level reaching the square's border, past which nothing was read, and a shift beside that region with no reading (a non-finite cell of the surface, which the overlap reading does not produce; see [above](#the-radius-the-slide-and-the-surface)). A shift with no reading whose read neighbours are all below the level is taken to be below it, as the radius takes it. `SelfSimilarity::contour` records both kinds of opening, and each measure applies them as follows.

- **A gap inside the square.** The shifts with no reading next to a shift at the level are gathered with every shift with no reading connected to them. If that set touches the square's border, the region could continue through it past the border in any direction, so every measure is `at_least`, both grid axes, the radius and the image radius, whatever their values. Otherwise any crossing it hides lies on a grid edge between one of its shifts and a read neighbour, so the set and its read neighbours bound it: a measure is `at_least` when one of those shifts gives a larger value than the readings did (`|c_k|` for a grid axis, `|c|`, capped at `r`, for the radius, `|J·c|`, or `r·σ_max(J)` for a shift beyond `r`, for the image radius). Each measure is convex, so its largest value on an edge is at one of the edge's ends.
- **Running off the square along an axis.** A shift at the level on the border whose neighbour one step along `x` (or `y`) lies past the border runs off along `x` (or `y`). It makes the reach along that axis `at_least`, the radius `at_least` (it reads `r` anyway), and the image radius `at_least`.
- **The other axis of a run-off.** The readings say nothing past the border, so the region could turn and grow along the other axis there. The reach along the other axis stays exact only where the region **holds its width** over its last two lines before the border: take the run of shifts at the level along the border line through the open shift, `a ..= b`, and its crossings at the two ends, interpolated as the contour's are; every shift of the line just inside, `a ..= b`, is at the level, and the run there, with its own end crossings, spans at least as far at both ends, to within `1e-4` grid px. A region that keeps or narrows its width is extrapolated to keep doing so. The allowance is for the kernels' `f32` sums: a ridge of uniform stripes, the same on every line in exact arithmetic, reads crossings that differ from line to line by about `1e-7` grid px, and more where the surface falls through the level shallowly. `1e-4` is a thousand times that, and a region widening by `1e-4` per line would need ten thousand lines past the border to widen by one grid px. One that widens, moves sideways (a ridge slanted across the square), or whose run ends cannot be read (it reaches the square's corner, or a run end has no reading beside it) makes the reach along that axis `at_least` as well. So a ridge along `x` that runs off at `±x` with an even width reads its width along `y` exactly, while the ridge `z = 1 − 0.5·(dy − 3·dx)²`, at the level only at `(0, 0)` and `(±1, ±3)` for `r = 3`, runs off at `(1, 3)` towards `+y` with nothing at the level on the line inside it, and its reach along `x`, about 1.01 from its crossings, is `at_least`: it continues to `(2, 6)`.
- **The cap.** The radius is `at_least` where it reaches `r`, the reading printed `3+`, and the image radius wherever a point lies at or beyond `r` from the centre and was brought in to it.

The reach along the patch's axes takes the grid axes' flags. The radius's value is the reading's radius whatever its flag.


## Rust API

The operation lives in [self_similarity/](../../../crates/sfmtool-core/src/patch/self_similarity/) (`mod.rs` for the API, `kernels.rs` for the scalar and AVX2 kernels, `overlap.rs` for the overlap reading, `contour.rs` for the contour and its reach, `tests.rs`), as `sfmtool_core::patch::self_similarity`. The bench reads it in [bench/evaluate.rs](../../../crates/sfmtool-core/src/bench/evaluate.rs), the keypoint localizer's member gate in [keypoint_localize.rs](../../../crates/sfmtool-core/src/patch/keypoint_localize.rs) (`member_self_similarity_radius`), cluster-patch refinement's member gate in [cluster_refine/mod.rs](../../../crates/sfmtool-core/src/patch/cluster_refine/mod.rs) (`member_zncc_self_similarity_radius`), and it is bound as `sfmtool._sfmtool.patches.zncc_self_similarity_parts` and `zncc_self_similarity_parts_stack` in [patches/self_similarity.rs](../../../crates/sfmtool-py/src/patches/self_similarity.rs).

```rust
/// The template spread, in grey levels, under which a channel carries no
/// texture.
pub const FLAT_FLOOR: f64 = 0.5;

/// How far a patch can slide over itself and stay indistinguishable from
/// its true position.
pub struct SelfSimilarityParams {
    /// How far the shifts searched reach along each axis, in tile pixels,
    /// and the largest radius read; a radius of `max_radius` reads "this
    /// far or further".
    pub max_radius: u32,
    /// ε: the ZNCC deficit two views of the same surface show from warp,
    /// blur and lighting, as a fraction.
    pub relative_tolerance: f64,
    /// n: the noise between two views, in grey levels.
    pub noise: f64,
}

/// One template's reading.
pub struct SelfSimilarity {
    /// How far from the centre the surface falls through `1 − tolerance` at
    /// its furthest, interpolated linearly along the grid edges of the square
    /// of shifts, 0 ..= max_radius, in tile pixels; `max_radius` when the
    /// region at the level runs off the square.
    pub radius: f64,
    /// The direction of the indistinguishable shifts, scaled by how
    /// strongly they line up; `[0, 0]` when there are none.
    pub slide: [f64; 2],
    /// The tolerance τ the template was judged by; infinite for a template
    /// with no textured channel.
    pub tolerance: f64,
    /// z(d) for every shift of the (2r + 1)² square, row-major from
    /// (−r, −r); 1 at the centre.
    pub surface: Vec<f64>,
}

/// A bitmap whole, its middle square and the nine cells of the ZNCC
/// grid's split, each read as its own template.
pub struct SelfSimilarityParts {
    pub whole: SelfSimilarity,
    pub middle: SelfSimilarity,
    pub grid: [[SelfSimilarity; 3]; 3],
}

/// A tile, planar: channel `c`'s row-major `width × height` plane at
/// `values[c * width * height ..]`. One to three colour channels.
pub struct PatchTile<'a> {
    pub values: &'a [f32],
    pub channels: usize,
    pub width: usize,
    pub height: usize,
}

impl<'a> PatchTile<'a> {
    /// The planar colour channels of an interleaved `height × width × C`
    /// patch, the layout a rendered bitmap arrives in, dropping a fourth
    /// channel (alpha): returns the planes and the colour channel count.
    pub fn planes_from_interleaved(
        patch: &[f32],
        width: usize,
        height: usize,
        channels: usize,
    ) -> (Vec<f32>, usize);

    /// Which samples of an interleaved `height × width × C` patch carry
    /// data: with a fourth channel, those whose alpha is above 0; without
    /// one, every sample (`None`).
    pub fn data_from_interleaved(
        patch: &[f32],
        width: usize,
        height: usize,
        channels: usize,
    ) -> Option<Vec<bool>>;
}

/// One template inside `tile`, read the overlap way: at each shift, only
/// the template samples whose moved sample lies inside the tile, and where
/// both carry data, are correlated. `data` is one flag per sample, row-major;
/// `None` means every sample carries data.
pub fn zncc_self_similarity_radius(
    tile: &PatchTile<'_>,
    data: Option<&[bool]>,
    template: [usize; 4],
    params: &SelfSimilarityParams,
) -> SelfSimilarity;

/// A whole `R×R` bitmap, its middle `R/4 .. R - R/4` and each cell of the
/// `R/3`, `R - R/3` split, each read the overlap way, the middle and the cells
/// taking their shifted windows from the rest of the bitmap where it reaches.
pub fn zncc_self_similarity_parts(
    bitmap: &PatchTile<'_>,
    data: Option<&[bool]>,
    params: &SelfSimilarityParams,
) -> SelfSimilarityParts;
```

The functions panic, with a message naming the sizes, on a malformed tile (no channels, more than three, or a value count that does not match), on `data` with the wrong number of flags, on an empty template or one that does not fit in the tile, and, for the parts, on a bitmap that is not square or is under `3 × 3`. These are caller errors: the bench, the gates and the bindings size the tile from the same parameters they pass.

**Why this shape.** The input is the bitmap being judged and nothing around it, so a radius depends on that bitmap alone and every caller computes the same number from it. The template is still a rectangle inside the tile, so a part of a bitmap can be read with the rest of it as its shifted windows, as the middle square and the cells are. The parts function exists because its eleven templates share one bitmap and one set of moments, and computing them together is several times cheaper than eleven calls (see [Implementation notes](#implementation-notes)). The result carries `tolerance` so a display or a test can say why a shift counted, and `surface` because the reading computes every `z(d)` anyway and a display can draw the surface the radius was read from. The surface is a `Vec` because its size follows `max_radius`. The parameters are a struct and not constants because a caller may need other values than the defaults, and the defaults of `ε` and `n` are not yet calibrated on real pairs of views (see [Open questions](#open-questions)).

```rust
use sfmtool_core::patch::self_similarity::{
    zncc_self_similarity_parts, PatchTile, SelfSimilarityParams,
};

// `rendered` is a 24 × 24 × 3 tile rendered for one view: every sample is data.
let (planes, channels) = PatchTile::planes_from_interleaved(&rendered, 24, 24, 3);
let tile = PatchTile { values: &planes, channels, width: 24, height: 24 };
let parts = zncc_self_similarity_parts(&tile, None, &SelfSimilarityParams::default());
if parts.whole.radius < 1.5 {
    // Matching locks onto this patch within one pixel, diagonals included.
}
```

The functions take the data flags as a separate slice rather than reading a fourth channel of the tile, because `PatchTile` holds colour planes only; `data_from_interleaved` derives the flags from an RGBA bitmap in the same layout `planes_from_interleaved` reads.

```rust
use sfmtool_core::patch::self_similarity::{
    zncc_self_similarity_parts, PatchTile, SelfSimilarityParams,
};

// `bitmap` is a stored 24 × 24 × 4 consensus row, alpha 0 where no view
// covered the sample.
let (planes, channels) = PatchTile::planes_from_interleaved(&bitmap, 24, 24, 4);
let data = PatchTile::data_from_interleaved(&bitmap, 24, 24, 4);
let tile = PatchTile { values: &planes, channels, width: 24, height: 24 };
let parts = zncc_self_similarity_parts(&tile, data.as_deref(), &SelfSimilarityParams::default());
let passes = parts.whole.radius <= 2.5; // NaN (no data) is judged by the caller
```

The contour and its reach ([contour.rs](../../../crates/sfmtool-core/src/patch/self_similarity/contour.rs)):

```rust
/// One contour point, in the grid's axes, grid px from the centre.
pub struct ContourPoint {
    pub offset: [f64; 2],
    /// `Some([ex, ey])` where the neighbour that way has no reading: the
    /// crossing lies past the point in that direction.
    pub open_towards: Option<[i8; 2]>,
}

impl ContourPoint {
    /// Whether the point is open: `open_towards.is_some()`.
    pub fn is_open(&self) -> bool;
}

/// A length and whether the true length may be larger ("this far or further").
pub struct BoundedLength { pub value: f64, pub at_least: bool }

/// The reach along the patch's `u` and `v`, `[u, v]`.
pub enum PatchAxisReach {
    Length([BoundedLength; 2]), // w = 1, the scene's world-space unit
    Angle([BoundedLength; 2]),  // w = 0, degrees
}

impl PatchAxisReach {
    /// `[u, v]`, whatever the unit.
    pub fn values(&self) -> [BoundedLength; 2];
}

impl SelfSimilarity {
    /// `None` for a reading with no data, or a surface that is not a
    /// `(2r + 1)²` square for an odd side.
    pub fn contour(&self) -> Option<SelfSimilarityContour>;
}

impl SelfSimilarityContour {
    pub fn points(&self) -> &[ContourPoint];
    pub fn max_radius(&self) -> usize;
    pub fn radius(&self) -> BoundedLength;            // value == SelfSimilarity::radius
    pub fn grid_axes(&self) -> [BoundedLength; 2];    // max |dx|, max |dy|
    pub fn image_radius(&self, jacobian: [[f64; 2]; 2]) -> Option<BoundedLength>;
    pub fn patch_axis_reach(
        &self,
        placement: &OrientedPatch,
        resolution: usize,
    ) -> Option<PatchAxisReach>;
}

/// The contour measured in every unit that can be computed for it.
pub struct SelfSimilarityReach {
    pub grid_radius: BoundedLength,
    pub grid_axes: [BoundedLength; 2],
    pub image_radius: Option<BoundedLength>,
    pub patch_axes: Option<PatchAxisReach>,
}

impl SelfSimilarityReach {
    pub fn read(
        reading: &SelfSimilarity,
        jacobian: Option<[[f64; 2]; 2]>,
        placement: Option<&OrientedPatch>,
        resolution: usize,
    ) -> Option<Self>;
}

```

The Jacobian the image radius is read through, and the singular values it is bounded by past the cap, live beside `WarpMap` in [camera/warp_map.rs](../../../crates/sfmtool-core/src/camera/warp_map.rs), as `sfmtool_core::camera::warp_map`, since the first is the geometry of the warp a tile is rendered through, laid out as `WarpMap::from_patch` steps the grid, and the second is general 2×2 linear algebra:

```rust
/// Image px per grid px at the centre of a tile rendered through `placement`
/// at `resolution`, `[[dx/dcol, dx/drow], [dy/dcol, dy/drow]]`; `None` where
/// a point beside the centre does not project.
pub fn patch_grid_jacobian(
    placement: &OrientedPatch,
    camera: &CameraIntrinsics,
    cam_from_world: &RigidTransform,
    resolution: usize,
) -> Option<[[f64; 2]; 2]>;

/// The two singular values of a 2×2 matrix, larger first.
pub fn singular_values_2x2(m: [[f64; 2]; 2]) -> [f64; 2];
```

`patch_grid_jacobian` projects through `CameraIntrinsics::project_homogeneous` ([image-warping.md](../camera/image-warping.md)), the projection with no test against the image's bounds that the localizer, the bench's steps and the viewer share. On an equirectangular camera it brings each `x` difference into `±π·fx`, half the panorama's `2π·fx` period, before averaging, so a patch straddling the seam behind the camera reads the same Jacobian as one in front. `singular_values_2x2` gives `image_radius` its bound past the cap, and Track View's *Zoom* column its zoom ([track-view.md](../../gui/track-view.md)). The warp map's per-pixel SVD and view selection's `affine_sigma_major` compute the same singular values in their own `f32` arithmetic, which they keep so their results do not change by a rounding.

The contour is a method of the reading rather than a field of it because it is read from the surface and the tolerance the reading already carries, so the kernels and the overlap reading do not compute it and a stored reading can be measured later. The radius is computed from the same points, so the two cannot drift apart. `SelfSimilarityContour` keeps the points once for the several measures read from them, and `SelfSimilarityReach::read` is the one call for a caller that has a reading, a view and a placement, as the bench has; a caller that wants only the per-axis bound on the point in the scene's world-space unit calls `contour()?.patch_axis_reach(placement, R)` with no camera at all. The reach is a separate struct from the reading because the units it needs, the camera and the placement, are not the reading's: the same reading of a stored bitmap has a grid reach and nothing else.

```rust
use sfmtool_core::camera::warp_map::patch_grid_jacobian;
use sfmtool_core::patch::self_similarity::{PatchAxisReach, SelfSimilarityReach};

// `parts` read a 24 × 24 tile rendered through `placement`.
let jacobian = patch_grid_jacobian(&placement, &camera, &cam_from_world, 24);
if let Some(reach) = SelfSimilarityReach::read(&parts.whole, jacobian, Some(&placement), 24) {
    if let Some(PatchAxisReach::Length([u, v])) = reach.patch_axes {
        // The point could sit up to `u.value` along u and `v.value` along v
        // from where it is, in the scene's world-space unit; "or further"
        // where `at_least`.
    }
}
```

## Theory

### Why a curvature is not enough

The classical score of how well a patch pins its keypoint is the curvature of this ZNCC surface at `d = 0`, read through the structure tensor `M = Σ_k w_k ∇I_k ∇I_kᵀ` over the window [[Harris & Stephens 1988](#references)]. Its weaker eigenvalue `λ₂` is the Shi–Tomasi score [[Shi & Tomasi 1994](#references)], and scaled by the photometric noise it gives a positional uncertainty `σ_pos = σ_noise / √λ₂`, the precision of a least-squares fit of the peak [[Förstner & Gülch 1987](#references)]. It answers "once a match is on the peak, how precisely can the peak be placed under a little independent noise", and it answers that correctly. sfmtool gates on the radius instead, because the curvature does not answer "will a match from a nearby start lock onto this position and no other", for three reasons.

- **A strong straight or slightly bent edge reads as sharp.** A 20-level edge that bends a little has a large `λ₂`, because the tensor sums the edge's contrast, so `σ_pos` is small (0.06 to 0.13 grid px on the edges in the sample below) while the patch matches itself above 0.95 ZNCC five to seven pixels along the edge.
- **8-bit rounding reads as texture.** A sky ramp of a few grey levels steps by one level every few pixels in every direction, and the tensor counts each step as a gradient.
- **Only the neighbourhood of the peak is seen.** A second peak, a repeat, or a ridge that bends further out is invisible to a quadratic fitted at the centre.

The radius reads the surface itself over the whole bounded window, so a ridge in any direction, a bend, and a second peak within `r` all count.

### The tolerance

Two photographs of the same patch never match exactly, so "indistinguishable" has to mean "no worse than a true match". Two parts of a true match's ZNCC deficit behave differently.

- **Noise.** Two views of a template channel with spread `s_c`, each carrying independent noise of `n` grey levels, correlate at about `s_c² / (s_c² + n²)`, a deficit of about `(n / s_c)²`, and the channel average's deficit is the mean of those. It is large for a faint template and negligible for a strong one, and it is what sends a sky or a patch of grain to the maximum: its spread is a few levels in every channel, so the noise term exceeds any deficit a one-pixel shift produces.
- **Mismatch between views.** A warp that is off by a fraction of a pixel, a change of blur, or a change of lighting changes each pixel by an amount that grows with the local gradient, so its deficit is a roughly fixed fraction of the patch's own contrast. That is `ε`. It is what gives a strong edge its long radius: the edge's deficit along its own length is a few hundredths per pixel, which a true match between two views already exceeds.

ZNCC's own normalization handles the second part and not the first, which is why a plain cutoff on ZNCC (`1 − z ≤ 0.05`) scores flat patches as locked: grain against itself falls off within one pixel. The `(n / s)²` term puts the patch's contrast back where it matters.

### A bounded radius in pixels

Keypoint lock-in is decided in the first few pixels: a patch that matches itself two or three pixels away will be pulled there by a match that starts nearby, and the exact distance past that does not change what to do with it. A bound of 3 keeps the window at 49 shifts and makes "3" an honest "3 or more". The ZNCC is computed at whole-pixel shifts only, which is enough to see whether the peak is unique, since a ridge or a flat surface shows at whole pixels by construction. The radius is then read between them, where the surface falls through the level, so that two patches whose furthest matching shift is the same whole pixel are told apart by how far past it they keep matching, and a patch that locks reads how sharp its peak is instead of 0.

The radius is a Euclidean length because what decides lock-in is how far from the true position a match could land, in any direction: a shift of `(2, 2)` is 2.83 px away, not 2. It is the furthest point of the isosurface, as the linear interpolation places it, and not the furthest indistinguishable shift, so that a shift that clears the level by a ten-thousandth, with every shift beyond it well under, reads its own distance rather than a whole pixel more.

The search covers the square `|dx|, |dy| ≤ r` because the kernels compute every shift of it, and a crossing on an edge between two shifts can only be placed where both are read. The square reaches `r` along the axes and `r√2` at its corners, so it does not search every direction equally far: an edge along a diagonal can be followed to 4.24 at `r = 3`, one along an axis only to 3. The radius is capped at `r` so that the same edge scores the same turned by any angle, and so that `r` keeps its one meaning, "matched itself at the edge of the search in some direction and may slide further": every border shift is at least `r` away, so a region at the level that reaches the border reads `r` whichever side it leaves by, and a crossing past `r` towards a corner reads `r` too. The cap is also what the gates rely on, since a bar of `r` or more turns nothing out. A second peak counts wherever it lies in the square, so a near-repeat in a corner, up to `r√2` from the centre, reads `r` as one on the axes does. At `r = 1` the square holds the diagonal shifts, so an edge at 45° reaches its corner and reads 1, as an edge along an axis does.

An exact repeat is also a repeat at every multiple of its step, so a pattern that repeats every `(1, 1)` matches itself at `(3, 3)` as well, on the square's border, and scores 3. On a pattern that matches itself at `(1, 1)` and stops matching by `(2, 2)`, the radius reads between 1.41 and 2, where the surface falls through the level past `(1, 1)`.

On the ground truths of seoul_bull (1,277 sightings) and kerry_park candidate `tk106` (2,289), evaluated on the bench at the track stage, reading the whole square's furthest crossing with the cap changes the radius of 30 and 131 sightings, compared with reading the disk and sending any indistinguishable shift in its outermost ring to `r`. All but one fall, from 3 to between 2.24 and 2.99 (median 2.47 and 2.60), and 15 and 41 of them move under the default bar of 2.5. The one that rises, from 2.19 to 3, is an edge whose matching shifts `(−1, −3)` and `(1, 3)` lie in the square's corners outside the disk. Over the 17 and 48 images, with the localizer's gate reading a tile with a ring of `r` px around the core, Add Image to Tracks at its default recovers 1,000 and 1,695 of the 1,235 and 2,225 known observations at the ground-truth pose, against 989 and 1,660 with the outer-ring rule. The overlap reading of kerry_park's 384 stored consensus bitmaps culls 102 at 2.5, against 116.

### Why only the bitmap is read

The overlap reading judges the bitmap and nothing past it. The other way to read the same `R × R` grid is as the template of a tile rendered `r` px wider, with the frame grown by `(R + 2r)/R`, so that the grid is the wider tile's middle and the moved windows read the pixels around it (a **ringed** reading). Both were taken for every observation of the checked-in ground truths, at the same keypoint and placement: the bench tile (`R = 24`, every sample as data); the localizer's view, its `R × R` grid at the seed offset in the context tile it renders at a search of 6; and the cluster grid, `sample_member_grid` (`R = 25`) through an affine shape taken from the anchored placement's Jacobian, since the ground truths carry no SIFT shapes. Differences are overlap − ringed, in grid px:

| | seoul_bull | kerry_park |
|---|---|---|
| Observations (points) | 1,277 (280) | 3,767 (391) |
| Bench tile: correlation | 0.997 | 0.995 |
| Bench tile: mean difference | −0.029 | −0.041 |
| Bench tile: 5th / 50th / 95th percentile | −0.13 / −0.01 / +0.02 | −0.18 / −0.02 / +0.03 |
| Bench tile: fail rate at 2.5, ringed / overlap | 10.7% / 10.3% | 17.1% / 14.9% |
| Bench tile: verdicts at 2.5 that differ | 10 (0.8%) | 113 (3.0%) |
| of which ringed fails and overlap passes | 8 | 98 |
| of which ringed passes and overlap fails | 2 | 15 |
| Bench tile: overlap bar with the ringed 2.5's fail rate | 2.42 | 2.38 |
| Localizer: correlation, verdicts that differ (ringed fails / overlap fails) | 0.997, 12 (8 / 4) | 0.996, 102 (90 / 12) |
| Cluster grid: correlation, verdicts that differ (ringed fails / overlap fails) | 0.998, 10 (8 / 2) | 0.996, 101 (90 / 11) |
| Middle square, bench tile | identical | 4 verdicts differ |

The overlap reading reads slightly shorter, because the whole bitmap and its edge and corner cells lose the moved rows and columns that go past the bitmap, and with them the part of a long ridge they would read. The two readings agree closely enough that the default bar of 2.5 means nearly the same on either: the overlap bar that fails as many observations of the bench tile as the ringed reading does at 2.5 is 2.42 on seoul_bull and 2.38 on kerry_park, and the bar with the fewest verdicts changed over both datasets at the bench tile, 2.4, still changes 96 of 5,044 (1.9%) against 123 (2.4%) at 2.5, since the differences are spread over the bitmaps rather than one shift of the scale. The localizer's and the cluster grid's verdicts differ about as often as the bench tile's. The middle square has its moved windows inside the tile either way, so it reads the same, except on kerry_park, where the `R × R` render and the middle of the wider render sample the fisheye photographs slightly differently.

### What the defaults rest on

On 1,350 patches from `seoul_bull_sculpture` and `kerry_park`, drawn evenly from four kinds (flat, faint texture, edge, texture or corner) sorted by measures that none of the scores use, plus crops around seoul_bull's matched keypoints, the radius with `ε = 0.05`, `n = 2` and `r = 3`, measured on a 12 × 12 template of the RGB crop, gives:

| Kind | under 1 | 1 to 2 | 2 to 3 | 3+ | median below 3 |
|---|---|---|---|---|---|
| flat | 0% | 0% | 0% | 100% | |
| faint texture | 16% | 22% | 9% | 53% | 1.28 |
| edge | 37% | 23% | 5% | 35% | 0.88 |
| texture or corner | 92% | 7% | 0% | 1% | 0.25 |
| matched keypoint | 96% | 3% | 0% | 1% | 0.32 |

Scoring luminance instead changes 91 of the 1,350 patches, 80 of them to a longer radius, because it cannot see structure that is in the colour alone. On luminance, without the noise term, a plain 0.95 cutoff scores 51% of the flat patches 0.

## Implementation notes

Each `z_c(d)` comes from five sums over the overlap, per channel and shift: `Σ t`, `Σ u`, `Σ t²`, `Σ u²` and `Σ t u`, with the count `n` of the overlap; the centred cross sum is `Σ t u − Σ t Σ u / n`, and the centred sums of squares likewise. These are invariant to a constant subtracted from `t` and `u` together, so each route centres the values as suits its precision. The cost is the cross sums `Σ t u`, one dot product of the template against the moved window per channel and shift; two routes compute them.

- **The dense route.** Where every sample carries data, which is every rendered tile and almost every stored bitmap, the overlap at each shift is a rectangle: the template's rows and columns whose moved sample is inside the bitmap. `Σ t` and `Σ t²` over it, and `Σ u` and `Σ u²` over the same rectangle moved by `d`, come from a summed-area table per channel in `f64`, of the tile centred by its own mean. The cross sums come from the kernel, run for each template over a copy of the tile `r` px around it with zeros past the tile's edge, so a moved sample past the edge reads 0 and adds nothing to `Σ t u`, and the kernel needs no clipping; 8 more zero columns per row keep its widest load inside the row. Each channel of the copy is centred by the template's own mean `a`, so the `f32` products are on the scale of the template's own spread however far its level is from the rest of the tile's, and the cross sum is taken back to the tile's centring in `f64` as `Σ t u = Σ t′ u′ + a (Σ t + Σ u) − n a²`. On a 24 × 24 colour bitmap with a fine texture whose halves stand 1000 or 60000 grey levels apart, the cells and templates that lie within one half match the masked route's surface to `1e-5`, as do those that hold both halves at a step of 1000. At a step of 60000 those that hold both halves are held to `1e-3`: they carry the step in their own `f32` values, which at a level of 30000 keep steps of 0.002, the scale of the bitmap's own `f32` resolution there. A centred sum of squares taken from the tables, the template's or the moved window's, that is at or under `64 ε √N S`, with `N` the tile's sample count and `S` the channel's sum of squares over the whole tile, is the tables' rounding rather than spread, and is set to exactly 0: a window constant at a level far from the tile's mean keeps a residue of a few `ε S`, about `1e-5` at a level of 60000 in a 24 × 24 tile, which the absolute `1e-6` flat test would take for texture. On a 24 × 24 colour bitmap textured from 0 to 25 left of `x = 12` and constant at 60000 right of it, a 14 × 14 template across the step, whose windows moved right lie wholly on the constant side, and a 2 × 2 template just left of the step, whose windows moved 2 or 3 px right lie on the constant side too, read the same radius and slide by both routes, with surfaces within `1e-4`. A copy is made per template the kernel runs on, ten for the parts, and costs about 3 µs of the parts' 36 µs at `R = 24`. The kernels accumulate raw sums `X(d) = Σ_k P(k) P(k + d)` of the copy `P` in `f32` over bands of at most 8 template rows, and each band's sums are added into `f64` totals, which bounds the `f32` rounding a large template accumulates. The combine into `z(d)`, the tolerance test and the slide's second moments run in `f64`. The radius is tracked as the largest squared length of a crossing, and its one square root is taken after the last shift.
- **Scalar reference.** For each template pixel, each shift row `dy` and each `dx`, accumulate `P(k) · P(k + d)`. It is the oracle for the AVX2 kernel, the fallback on other CPUs and architectures, and the path for `2r + 1 > 8`.
- **AVX2.** Lanes run across the `2r + 1 ≤ 8` horizontal shifts, so one 8-lane register holds a whole shift row of the square of shifts, and `2r + 1` registers hold the window (7 at `r = 3`); the kernel is monomorphized on that count, so the accumulators stay in registers. Per template pixel it broadcasts `P(k)` and, for each shift row, loads the 8 plane values starting at `k + (−r, dy)` and fuses a multiply-add. The lanes past `2r + 1` are computed and discarded: masking them would cost more than the multiply-adds it saves. The padding columns keep every load inside its row's storage, and the bounds that guarantee it are asserted before the kernel runs. It is dispatched at run time on `is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")`, as the localizer's search kernels are ([keypoint-localization-search-cache.md](keypoint-localization-search-cache.md)), and compiled only on `x86_64`. The two kernels agree to `f32` rounding, not bit for bit, since the scalar one rounds each product before adding it.
- **Parts together.** The nine cells tile the bitmap and every sum adds, count included, so the parts function computes each cell's sums at every shift and adds them into the whole bitmap's. The middle square is not a union of cells and gets its own pass. That is about 35,000 multiply-adds per channel per bitmap at `R = 24`, `r = 3`, about 105,000 for a colour bitmap. In a release build on one core, a whole reading of a 24 × 24 colour tile takes about 8 µs and the parts about 36 µs, most of it in the summed-area lookups and the eleven combines rather than the cross sums. The render of the tile that the bench and the localizer read is not measured separately.
- **The masked route.** Where some sample carries no data, the overlap is not a rectangle and the summed-area tables do not give its moments, so for each shift it walks the samples of each part directly and accumulates the five sums per channel over the samples where both sides carry data, in `f64`, with each channel first centred by its mean over the samples that carry data. That is about 720 samples per shift at `R = 24`, 49 shifts, and 16 sums per sample for a colour bitmap: roughly 0.6 ms of one core per bitmap, and about 20 µs per bitmap of wall time for a stack of 17,664 on 32 cores. It has no AVX2 kernel: a mask in the inner loop does not map onto the unmasked cross-sum kernel. Data flags that are all set take the dense route.

## Where it runs

The [keypoint localizer](patch-keypoint-localization.md#the-member-self-similarity-gate) reads the whole radius of each view's own `R × R` core at its seed offset, cut from the context tile it has already rendered and read the overlap way, so no pixel of the tile around the core enters it, and its member gate, `KeypointLocalizeParams::max_member_zncc_self_similarity_radius`, drops a view whose radius is over the bar. A `NaN` radius fails the gate, `0` turns it off, and the default is `2.5`, chosen by the user from a sweep on seoul_bull and kerry_park ([The member gate's default](patch-keypoint-localization.md#the-member-gates-default)). [Add Image to Tracks](../reconstruction/add-image-to-tracks.md) reads the same radius for the new view's core at the point's projection, reports it per candidate as `zncc_self_similarity_radius`, and refuses a candidate over the same gate as `unlocalizable`.

[Cluster-patch refinement](cluster-patch-refinement.md#which-member-anchors-and-which-members-are-eligible) reads the whole radius of each member's own `R × R` template grid at its SIFT seed geometry (`sample_member_grid`), the overlap way (`member_zncc_self_similarity_radius`). Its member gate, `ClusterRefineParams::max_member_zncc_self_similarity_radius` (`sfm cluster-patches --max-member-zncc-self-similarity-radius`), refuses a member over the bar as `rejected_unlocalizable` before reference selection, with the localizer's pass rule and its default of `2.5`.

Two batch culls read a point's **consensus bitmap** by the [overlap reading](#the-overlap-reading), through `sfmtool._sfmtool.patches.zncc_self_similarity_parts_stack` and the shared Python pass rule `points_passing_zncc_self_similarity_radius` in [_filter_by_zncc_self_similarity_radius.py](../../../src/sfmtool/xform/_filter_by_zncc_self_similarity_radius.py): `sfm embed-patches --max-zncc-self-similarity-radius` (`embed_patches(max_zncc_self_similarity_radius=)`) on each point's round-1 consensus, before round 2 ([embed-patches-command.md](../../cli/reconstruction/embed-patches-command.md)), and `sfm xform --filter-by-zncc-self-similarity-radius` (`FilterByZnccSelfSimilarityRadiusTransform`) on a reconstruction's stored `patch_bitmaps` ([xform-command.md](../../cli/reconstruction/xform/xform-command.md)). A point passes at or under the bar and fails on a `NaN` radius; a bitmap with no sample carrying data has no reading and passes, as the bench's painting passes a row with no reading, so a point with no consensus is left to the pipeline's other rules. `0` turns the cull off, `3` or more turns nothing out, and the default is `2.5`, the member gates' `DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS`, which the bindings export under that name. At the default, on embed-patches runs over seoul_bull and kerry_park measured under the earlier disk rule (a radius of `r` for any indistinguishable shift in the outermost ring of the disk `dx² + dy² ≤ r²`), the embed-patches cull drops 59 of 837 points on seoul_bull and 292 of 1,886 on kerry_park; the xform filter over those runs' stored bitmaps, written with the cull off, removes 69 of 765 and 316 of 1,795.

The member gates and the consensus culls ask different questions, and neither makes the other redundant: a point whose members each pin a position can still fuse a consensus that slides, and a point with one textureless member can still fuse a sharp consensus from the others. So the member gates judge each view's own tile before it votes, and the culls judge the fused bitmap.

The `unlocalizable` refusal, the `.matches` member status `rejected_unlocalizable` and the localizer's `N_DROP_UNLOCALIZABLE` counter name what was refused, a view whose position a match cannot pin, rather than the score that judges it. Files written before these gates read the radius carry `max_keypoint_uncertainty` in a `.matches` file's `refine_options` or an `.sfmr`'s `tool_options`; nothing reads those keys back.

The bench computes it at both stages, for every observation that has a pixel (see [editable-track.md](../bench/editable-track.md) § "The ZNCC self-similarity radius"):

- **Track stage.** The tile is the `R × R` patch rendered through the keypoint-anchored frame, the grid a stored patch bitmap is rendered on, read the overlap way with every sample as data.
- **Cluster stage.** The tile is the one cluster refinement's member gate reads, the member's `R × R` grid at its seed geometry (`sample_member_grid`). The bench's refinement runs with that gate off, so the bench's whole radius is the number the gate would have judged.

The bench measures the [reach](#the-contour-and-its-reach) of both readings, the whole and the middle, at the track stage through the placement it renders the tile through, the track's placement re-anchored on the observation's keypoint (`OrientedPatch::anchored_at_keypoint`, falling back to the placement itself where the keypoint's ray cannot meet it), at `resolution = R`. `R` is the reconstruction's patch resolution, the edge of its patch bitmaps or else the evaluation's 24 ([editable-track.md](../bench/editable-track.md)). `J` is `camera::warp_map::patch_grid_jacobian` of that placement at `R`: the finite difference across the four points half a grid px either side of the centre, the four middle texel centres of the warp map at `R`, each projected with no test against the image's bounds (`CameraIntrinsics::project_homogeneous`), so a tile partly or wholly off the photograph still has one. At the cluster stage the tile is the member grid at the seed shape `S`, an affine map, so `J = S · 2·radius/R`, and there is no patch, so no reach along its axes.

Both read the default `SelfSimilarityParams`. The measurements carry `zncc_self_similarity_radius`, `zncc_self_similarity_radius_middle`, `zncc_self_similarity_radius_grid` (`[[f64; 3]; 3]`), `zncc_self_similarity_slide_grid` (`[[[f64; 2]; 3]; 3]`), `zncc_self_similarity_surface` (the whole tile's surface, `Vec<f64>`), `zncc_self_similarity_tolerance` (the whole tile's tolerance, `None` where it is flat), and `zncc_self_similarity_reach` and `zncc_self_similarity_reach_middle` (the whole tile's and the middle's `SelfSimilarityReach`, measured as [The contour and its reach](#the-contour-and-its-reach) says, with no `patch_axes` at the cluster stage), each `None` where the tile could not be rendered or sampled. The bench's `Thresholds::max_zncc_self_similarity_radius` bar judges the whole tile's radius, and the threshold painting turns out a row whose radius is over it: a patch that slides over itself that far and still matches, such as a straight edge or a flat patch, does not pin its position. Its default, `BENCH_MAX_ZNCC_SELF_SIMILARITY_RADIUS`, is 2.5 patch-grid px, the same as the localizer's member gate, under the largest shift searched; since the radius reads at most `r`, a bar of `r` or more turns nothing out. A row with no reading clears the bar and a `NaN` fails it. No bar judges the middle, the grid, the slide or the surface.

**Track View** has a *Self-similarity* column, reading the whole radius over the middle one to one decimal (`0.4 px whole` over `1.4 px mid`, with `3+ px` for the maximum) and drawing the grid: green under 1, yellow from 1 to 2, orange from 2 to under `r`, red at `r` or more, with a line along the slide in a cell whose slide is at least 0.5 long, a hover on the numbers that tabulates the whole and middle reach in grid px, image px and along the patch's `u` and `v` in the file's world-space unit (or degrees for a patch at infinity), and beside the grid the tile's surface plot: the surface interpolated between the shifts and drawn as a heatmap, with the contour at `1 - tolerance` over it and the shifts inside it marked ([track-view.md](../../gui/track-view.md)), and a *self-sim. px* box that sets the bar, from 0 to `r` to one decimal. `get_bench_track` reports the same fields in both blocks, the reach as nested objects with each length as `{value, at_least}` and `patch_axes` carrying the reconstruction's `world_space_unit`, and the bar in its `thresholds`, which `apply_bench_track_thresholds` sets, the grids as three rows of three and the surface as rows of numbers with null where the tile is flat ([mcp-server.md](../../gui/mcp-server.md)). The Python observation dicts carry them as floats, float64 `(3, 3)` and `(3, 3, 2)` arrays, a `(2r + 1, 2r + 1)` float64 surface, and the reach as nested dicts in the wire's shape, with `patch_axes` as `{"kind", "along"}` and no unit, since the dict does not carry the reconstruction; the bar is in `EditableTrack.thresholds` and is a keyword of `apply_thresholds`.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `max_radius` | `3` | How far the shifts searched reach along each axis, in tile pixels; the radius is capped here. |
| `relative_tolerance` | `0.05` | ε, the ZNCC deficit a true match between two views shows from warp, blur and lighting. |
| `noise` | `2.0` | n, the noise between two views, in grey levels. |
| `FLAT_FLOOR` | `0.5` grey levels | A channel whose template spread `s` is under this has no texture; a template with no channel at or above it scores `max_radius`. |

The three parameters' defaults are defined on `SelfSimilarityParams::default()`, and the flat floor is the constant `FLAT_FLOOR` in [self_similarity/mod.rs](../../../crates/sfmtool-core/src/patch/self_similarity/mod.rs).

## Python bindings

`sfmtool._sfmtool.patches.zncc_self_similarity_parts(bitmap, *, max_radius=3, relative_tolerance=0.05, noise=2.0)` reads one `(R, R)` single-channel bitmap or `(R, R, C)` patch, uint8 or float32, by the [overlap reading](#the-overlap-reading), a fourth channel read as the data flags (alpha above 0), and returns a dict of `radius` (float), `radius_middle` (float), `radius_grid` (`(3, 3)` float64), `slide` (`(2,)` float64), `slide_grid` (`(3, 3, 2)` float64), `tolerance` (float) and `surface` (`(2r + 1, 2r + 1)` float64), the last three of the whole bitmap; a bitmap with no sample carrying data has no reading and its values are `NaN`. It does not return the reach, since a bitmap alone has no camera and no placement to measure it in; only the bench's observation dicts carry it. A bitmap of another dtype or rank, not square or under `3 × 3`, or with more than four channels, raises `ValueError`. The bench fields arrive through the existing observation dicts.

```python
from sfmtool._sfmtool.patches import zncc_self_similarity_parts

out = zncc_self_similarity_parts(bitmap_24x24x3_uint8)
out["radius"], out["radius_grid"], out["surface"][3, 3]  # 1.0 at the centre
```

`sfmtool._sfmtool.patches.zncc_self_similarity_parts_stack(bitmaps, *, max_radius=3, relative_tolerance=0.05, noise=2.0)` reads each bitmap of an `(N, R, R, C)` uint8 or float32 stack the same way, in parallel over the bitmaps, a fourth channel read as the data flags (alpha above 0), and returns a dict of per-bitmap arrays: `radius` (`(N,)` float64), `radius_middle` (`(N,)`), `radius_grid` (`(N, 3, 3)`), `slide` (`(N, 2)`), `tolerance` (`(N,)`), the whole bitmap's where not said otherwise, and `covered` (`(N,)` bool), false for a bitmap with no sample carrying data, whose values are `NaN`. A stack of another dtype or rank, of bitmaps that are not square or under `3 × 3`, or with more than four channels, raises `ValueError`. It is a module function rather than a `PatchCloud` method because it reads the bitmaps alone: no geometry, no views. `DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS` (`2.5`) is exported beside it.

```python
from sfmtool._sfmtool.patches import zncc_self_similarity_parts_stack

out = zncc_self_similarity_parts_stack(recon.patch_bitmaps)
passes = ~out["covered"] | (out["radius"] <= 2.5)
```

## Testing

In [self_similarity/tests.rs](../../../crates/sfmtool-core/src/patch/self_similarity/tests.rs):

- The scalar and AVX2 kernels agree on every shift's ZNCC within `1e-4` over random rough and smooth tiles, on the radius within `1e-3` (it is interpolated from the ZNCC, so it carries their rounding, amplified where the values either side of a crossing are close), and on the slide and tolerance exactly, at `r` from 1 to 3, templates from 4 × 4 to 24 × 24 inside the tile and in its corner, where the moved windows run off it, and one to three channels (`avx2_matches_scalar`). The dense and masked routes agree on the whole bitmap, its middle, its cells and templates on its edges, and flags that are all set take the dense route (`the_dense_and_masked_routes_agree`). On a bitmap whose halves stand 1000 or 60000 grey levels apart, each with a fine texture, they agree to `1e-5` in the surface for every template within one half, and to `1e-3` for a template holding both levels at 60000, whose own `f32` values carry the step (`the_dense_route_keeps_its_precision_across_a_large_step`).
- A red-and-green edge of equal luminance scores as an edge, not as flat; a channel flat in the template is left out; a grey tile repeated in three channels scores as the single channel does.
- A strong corner and a blob score under 1 and more than 0; a straight edge at 0°, 30°, 45° and 90° scores `r` at `r = 2` and `r = 3`, with its slide along the edge; a flat tile, a tile of pure 1-level noise, and an 8-bit sky ramp score `r`; a pattern repeating every 2 px along one axis scores between 2 and 2.3, and one that matches itself at `(1, 1)` but not at `(2, 2)` between 1.41 and 2.
- The radius is where the ZNCC, interpolated linearly along a grid edge, equals the level, at the furthest such point; a real sighting's surface whose one shift at the level, `(−2, 1)`, clears it by 0.0002 reads about √5 = 2.24; a ridge along the diagonal reads its crossing just past `(1, 1)`; a region at the level that runs off the square, along an axis or a diagonal or through one border shift, reads `r`; and a shift at the level beside a non-finite reading counts its own distance there.
- The surface is 1 at its centre and finite over the whole square.
- The parts function agrees with separate calls on each part's template.
- A template that does not fit in the tile, and a parts bitmap that is not square, are refused with a panic message naming the sizes.
- The overlap reading: a corner locks (under 1); a straight edge at 0°, 30°, 45° and 90° reads 3 with its slide along the edge; a flat bitmap and an 8-bit ramp read 3; samples without data drop out, so what they hold does not change the reading, which equals that of the covered columns cut out on their own, and a cell with no data, or a bitmap with none, has no reading; on a bitmap cut from a larger textured tile, the middle and the centre cell read as the same templates read inside the larger tile, over the whole square of shifts, and the whole bitmap within 0.25 of it, with the same tolerance; the parts agree with separate calls; the wrong number of data flags is refused; and `data_from_interleaved` reads alpha.
- The contour: on every fixture above, templates with tile around them and whole bitmaps, the flat template included, its radius is the reading's radius to the bit; a ridge along `x` reaches 1.5 along `x` and 0.1 along `y`; through `J = diag(2, 1)` its image radius doubles and through `diag(1, 2)` it does not; along `u` and `v` it scales by each half-extent, `0.5` and `2`, and on a patch at infinity reads `atan` of that in degrees; a ridge running off the square with an even width is a lower bound along that axis and exact along the other, as is a tile of uniform stripes read by the kernels, whose crossings differ from line to line by `f32` rounding, while one that widens towards the border, a lone border shift at the level and the slanted ridge above are lower bounds along both axes (and a wider search shows the slanted one reaching 2 along `x`), and a flat template is a lower bound everywhere; the image radius is a lower bound under `J = diag(1, 0.1)` where an open point maps short of a crossed one; a gap with no reading beside the region makes each measure a lower bound where its shifts reach further than the value and leaves it exact where they do not, and one that reaches the border makes every measure a lower bound; a reading with no data has no contour; and `patch_grid_jacobian` is diagonal at the patch width over `R` on a fronto-parallel patch, agrees with the warp map's four middle texels on a slanted one, is `None` behind a pinhole camera, and on a 2000 × 1000 equirectangular camera reads the same Jacobian for a patch straddling the seam behind it as for one in front ([warp_map/tests.rs](../../../crates/sfmtool-core/src/camera/warp_map/tests.rs)).

The bench's track and cluster evaluations fill every field for each observation with a pixel, the painting turns out a row whose radius is over the bar, keeps a row with no reading, and turns out nothing with the bar at the largest radius searched, and the default bar sits under that radius ([bench/tests.rs](../../../crates/sfmtool-core/src/bench/tests.rs)); the track stage's readings are the overlap readings of the `R × R` tile rendered through the anchored placement, and the cluster stage's radius is the member gate's; both fill the reach, its image radius equal to the contour's through `patch_grid_jacobian` of the anchored placement at the track stage and through `S · 2·radius/R` at the cluster stage, and its patch axes the grid axes times `2·half_extent/R` at the track stage; the viewer's tests cover the column's text, colours, marks, hover and heading ([track_view/body/tests.rs](../../../crates/sfm-explorer/src/track_view/body/tests.rs)) and the wire fields ([mcp/tests/bench.rs](../../../crates/sfm-explorer/src/mcp/tests/bench.rs)); [test_self_similarity_rust_bindings.py](../../../tests/rust_bindings/test_self_similarity_rust_bindings.py) covers the binding, and [test_bench_rust_bindings.py](../../../tests/rust_bindings/test_bench_rust_bindings.py) reads the reach from an evaluated observation's dict. The localizer's tests ([keypoint_localize/tests.rs](../../../crates/sfmtool-core/src/patch/keypoint_localize/tests.rs)) check that its member gate drops a flat and an edge view and keeps textured ones at a bar of 2, drops nothing at `0` or at `r`, gives the same verdicts at a search of 1 as at 6, reads the same radius for a core whose surrounding tile is filled with noise as for a clean one, passes at or under the bar and fails `NaN`, is on at 2.5 by default, and that a reference search reports the radius; [test_add_image_to_tracks_rust_bindings.py](../../../tests/rust_bindings/test_add_image_to_tracks_rust_bindings.py) checks that a bar refuses as `unlocalizable` exactly the candidates whose radius is over it. Cluster refinement's tests ([cluster_refine/tests.rs](../../../crates/sfmtool-core/src/patch/cluster_refine/tests.rs)) check that a textured member reads well under the bar and a flat, an edge and a smooth member over it, that its radius is the overlap reading of the member grid, that an edge whose surroundings past the grid hold the same edge inverted reads the same radius as the edge alone, while a reading of the grid with a ring of 3 samples around it reads under 3, that at the default a flat and an edge member are refused and a textured one kept, that `0` and `3` refuse nobody, and the pass rule. [test_filter_by_zncc_self_similarity_radius.py](../../../tests/xform/test_filter_by_zncc_self_similarity_radius.py) checks the overlap binding on a flat, an edge, a textured and an empty bitmap, the shared pass rule (the default culls the flat and the edge point and keeps the textured and the empty one, `0` and `3` keep every point), and the xform filter on those bitmaps and on real ones; [test_embed_patches_command.py](../../../tests/patch/test_embed_patches_command.py) checks the embed-patches cull on the same four kinds, its default of 2.5, and that its flag reaches `embed_patches` and the written `tool_options`.

## Non-goals

- It does not read pixels outside the bitmap it judges, even where a caller could render them.
- It does not measure sub-pixel precision, which the sub-pixel refinement covers.
- It does not change any file format.

## Open questions

- **Calibrating the defaults of ε and n on real pairs of views.** `relative_tolerance` and `noise` are parameters; their defaults in `SelfSimilarityParams::default()` rest on the single-image sample above and are not checked against matching between two views. The check that would set them starts each sighting's search on the ground-truth tracks of `seoul_bull_sculpture` and `kerry_park` 1, 2 and 3 px off its true keypoint along each axis and along the diagonals, and records whether the localizer's ZNCC search walks back. A sighting's measured lock-in is the largest offset it recovers from, and the radius should predict it: a sighting scored under 1 recovers from every offset up to 3, and one scored 3 fails from some. Scoring the curvature at the peak on the same sightings would show whether it predicts lock-in any better, and the comparison decides the default tolerance, the bench's default bar, and whether the batch kernels' gates change. The residuals between views that the track stage already computes are a direct measurement of `n` for it.
- **The unit of a per-view bar.** On seoul_bull and kerry_park a fifth and a third of the localizer's views read the largest radius, and against the current ground truths those views land as often within 1 px as views reading under 1, so the localizer's default bar of 2.5 costs recall there ([The member gate's default](patch-keypoint-localization.md#the-member-gates-default)). On seoul_bull the sightings over 2.5 are mostly smooth bronze seen where the patch covers few image pixels (48.6% of sightings read over 2.5 at a texel scale under 0.5 image px per patch-grid px, about 1% at 0.75 and above), which the user reads as patches that should have been bigger. The radius is in patch-grid px; where the grid is finer than the source pixels it samples, the rendered tile is smooth and matches itself a few grid px away however well the source pins it. Whether a radius read in source px, or on a grid no finer than the source, separates the views that register from those that do not is not yet measured.
- **Sharing the localizer's search kernel.** `keypoint_localize::kernels::compute_channel_grids` already computes cross sums and window moments over a dense shift grid for a weighted template. A self-similarity call is that kernel with the tile's own core as the template; sharing it would mean lifting it out of `keypoint_localize` and giving it an unweighted form.

## References

- C. Harris and M. Stephens, "A Combined Corner and Edge Detector," 4th Alvey Vision Conference, 1988, pp. 147–151. Defines the second-moment matrix `M = Σ w ∇I∇Iᵀ`, whose eigenvalues are "proportional to the principal curvatures of the local auto-correlation function". <https://bmva-archive.org.uk/bmvc/1988/avc-88-023.pdf>
- J. Shi and C. Tomasi, "Good Features to Track," IEEE CVPR 1994, pp. 593–600. Accepts a window as a feature when the smaller eigenvalue of that matrix is over a bar. Also Cornell TR 93-1399: <https://users.cs.duke.edu/~tomasi/papers/shi/TR_93-1399_Cornell.pdf>
- W. Förstner and E. Gülch, "A Fast Operator for Detection and Precise Location of Distinct Points, Corners and Centres of Circular Features," ISPRS Intercommission Conference on Fast Processing of Photogrammetric Data, Interlaken, 1987, pp. 281–305. Gives the precision of the located point as the inverse of that matrix scaled by the noise. <https://cseweb.ucsd.edu/classes/sp02/cse252/foerstner/foerstner.pdf>
