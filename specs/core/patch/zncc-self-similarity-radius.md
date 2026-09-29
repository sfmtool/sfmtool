# ZNCC Self-Similarity Radius

A keypoint is only worth keeping if matching can find it again: when the patch around it is compared with another photograph, the best match should sit at one place and nothing else nearby. The ZNCC self-similarity radius measures that from a single image. It slides the patch over itself by whole pixels, up to a small maximum distance, and asks how far it can move while still looking as much like itself as a true match between two views would. A corner or a busy texture stops matching itself within a pixel and scores 0 or 1. A straight edge keeps matching itself along its own length and scores the maximum, and so does a flat sky, whose only texture is noise and rounding. Alongside the distance, the score reports the direction the patch can slide in, and the patch's ZNCC with itself at every shift it tried. It is computed for the whole patch, its middle, and each ninth of it, like the bench's other patch readings.

It is the score sfmtool judges a keypoint's position by. The classical score, the curvature of the same surface at its peak, reads a strong straight edge as sharp, and is not used ([Why a curvature is not enough](#why-a-curvature-is-not-enough)). The bench judges the radius, with its `max_zncc_self_similarity_radius` bar. The keypoint localizer's member gate, `max_member_zncc_self_similarity_radius`, Add Image to Tracks, which uses the same gate, and cluster-patch refinement's member gate, `ClusterRefineParams::max_member_zncc_self_similarity_radius`, also judge the radius, with a default bar of 2.5 patch-grid px, the bench's default bar too. So do the two culls on a point's stored consensus bitmap, `embed-patches --max-zncc-self-similarity-radius` and `xform --filter-by-zncc-self-similarity-radius`, which read a bitmap that has no ring around it by [the overlap reading](#the-overlap-reading).

## What it measures

Take a tile with channels `I_c` and a template rectangle `Ω` inside it, with at least `r` pixels of the tile on every side of `Ω`, where `r` is the maximum radius. For every whole-pixel shift `d = (dx, dy)` in the disk `dx² + dy² ≤ r²`, compare the template with the window of the same size moved by `d`, one channel at a time:

```
z_c(d) = Σ_k t̃_c(k) · (I_c(k + d) − m_c,d)  /  sqrt( Σ_k t̃_c(k)² · Σ_k (I_c(k + d) − m_c,d)² )
z(d)   = mean of z_c(d) over the template's textured channels
```

The sums run over the pixels `k` of `Ω`, `t̃_c = I_c − mean_Ω(I_c)` is the centred template in channel `c`, and `m_c,d` is that channel's mean over the moved window. This is the ZNCC the pipeline matches with, per channel and averaged, with every pixel weighted equally, and `z(0) = 1`. Its channel conventions are the matching ZNCC's: a channel whose template spread `s_c` is under the flat floor is left out of the average, and a moved window with no variance in a channel (a centred sum of squares under `1e-6`, the matching ZNCC's flat constant) scores 0 in that channel, since a flat window is plainly different from a textured template. Colour is what matching sees, so a red-and-green edge of equal brightness, which is flat in luminance, is scored as the edge it is. A fourth channel (alpha) is not a colour and is not read.

A shift is **indistinguishable** from the true position when its ZNCC deficit is within the tolerance:

```
1 − z(d) ≤ τ ,     τ = ε + mean_c (n / s_c)²
```

where `s_c` is the template's own standard deviation in channel `c` in grey levels, the mean runs over the same textured channels as `z`, `n` is the noise between two views in grey levels, and `ε` is the relative mismatch between two views of the same surface (see [Theory](#theory)). `ε` and `n` are parameters of the score, passed in `SelfSimilarityParams` and the same for every patch in a call, so the radius is a property of the patch alone: it depends on the tile and the parameters, and on nothing about the views or the track the patch belongs to.

The **radius** is how far from the centre the surface falls through the level `1 − τ`. Over every grid edge between two neighbouring shifts of the disk, one at or above the level and the other below it, the ZNCC is interpolated linearly along the edge to the point where it equals the level; the radius is the largest distance of those points from the centre, tracked as a squared length with one square root at the end. The centre is always at or above the level, so a patch that locks reads the fraction of a pixel its peak takes to fall through it rather than 0. When an indistinguishable shift lies in the disk's outermost ring, `(r − 1)² < dx² + dy² ≤ r²`, the patch still matched itself at the edge of the window and may slide further, so the radius is `r`, read as "`r` or more". Every direction saturates at the same value this way. A template with no channel at or above the flat floor has no texture to match and scores `r`, with a slide of `[0, 0]`, an infinite tolerance and a surface of `NaN`.

The **slide** is the direction of the indistinguishable shifts. With `C = Σ d dᵀ / |A|` over the set `A` of indistinguishable shifts and eigenvalues `μ₁ ≥ μ₂`, the slide is the unit eigenvector of `μ₁` scaled by `1 − μ₂ / μ₁`, in the grid frame (`x` column-right, `y` row-down). A set strung out along a line gives a slide near unit length along it, a set spread evenly around the centre gives one near zero, and an empty set gives `[0, 0]`. Its sign means nothing.

The **surface** is `z(d)` itself over the `(2r + 1)²` square of shifts, row-major from `(dx, dy) = (−r, −r)`, with `1` at the centre and `NaN` for the shifts outside the disk. It is what the radius and the slide are read from, returned so a display can draw it.

## The overlap reading

The reading above needs `r` pixels of the tile around the template for the moved windows to read. A caller that renders the patch renders it wider, but a point's stored consensus bitmap (the `.sfmr` `patch_bitmaps_y_x_rgba` row, or the round-1 consensus `embed-patches` fuses) is exactly `R × R`, and the pixels past its edge were never rendered. The **overlap reading** scores such a bitmap as it is, using the full bitmap as the template: at each shift `d`, only the samples that have data on both sides are correlated.

- **The overlap.** At shift `d` the overlap `O_d` is the set of samples `k` of the template where `k` and `k + d` both lie inside the bitmap and both carry data. The per-channel ZNCC `z_c(d)` is the formula above with every sum over `O_d` instead of `Ω`, and with the template's mean and spread, and the moved window's, both taken over `O_d`, so every `z(d)` is a proper ZNCC on the same scale as the ringed reading's. At `d = 0` the overlap is the template's samples that carry data.
- **Data.** A sample carries data when its alpha (the fourth channel) is above 0; a bitmap with no fourth channel has data at every sample. The consensus fuse writes alpha 0 where no view covered the sample, where only one view did, and where the views disagree so much that the confidence rounds to 0, and its colour there is zero or one view's reading, so all of those drop out, as a sample off the edge does. A row for a point with no consensus is zero throughout, alpha included.
- **Channels.** The channels judged are those whose spread over the template's samples with data (`O_0`) is at or above the flat floor. At each other shift a channel whose template spread over `O_d` falls under the floor is also left out of that shift's average, and a moved window with no variance in a channel scores 0 in it, as in the ringed reading. A shift whose overlap leaves no channel, or has no sample at all, scores 0: nothing there matches the template.
- **The tolerance.** `τ` uses the spread `s_c` of the whole template, over `O_0`, not the spread over each shift's overlap. `τ` is the deficit a true match of this template between two views would show, which is a property of the template at its true position, where the overlap is the whole template. It also has to be one level for the whole surface: the radius is read where the surface, interpolated between neighbouring shifts, falls through `1 − τ`, and a level that changed from shift to shift would have no single crossing to interpolate. The overlaps are at least `(R − r)/R` of the template on each axis, so their spreads differ from the template's by a few per cent on a textured bitmap.
- **The rest is the ringed reading's.** The radius is read from the surface by the same linear interpolation along grid edges, an indistinguishable shift in the outer ring reads `r` or more, the slide and the surface are formed the same way, and a template with no textured channel at `d = 0` scores `r` with an infinite tolerance. A template with no sample carrying data has **no reading**: its radius, slide, tolerance and surface are all `NaN`.
- **The parts.** The middle square and the nine cells use the same rule, with the rest of the bitmap as their ring: at each shift a cell's sample counts when its moved sample lies inside the bitmap and both carry data. Where the moved window stays inside the bitmap, which is always the case for the middle (a margin of `R/4 ≥ r` at `R ≥ 12`) and the centre cell (`R/3 ≥ r`), and every sample carries data, the overlap is the whole part and the reading is exactly the ringed one; an edge or corner cell loses the rows and columns its moved window would read past the bitmap. One rule covers both cases, so there is no separate ringed path inside the bitmap.

On the patches it was checked against, the overlap reading of the whole bitmap reads slightly shorter than the ringed reading of the same core. On embed-patches runs over the seoul_bull and kerry_park solves (765 and 1,795 points), each point's patch was rendered at `R + 2r = 30` with its frame grown by `30/24`; the ringed reading of that tile's `24 × 24` core was compared with the overlap reading of the same core cut out, and with the overlap reading of the file's own stored `24 × 24` bitmap:

| | seoul_bull | kerry_park |
|---|---|---|
| Correlation, ringed vs overlap, same core (every sample as data) | 0.982 | 0.987 |
| Mean overlap − ringed | −0.050 | −0.052 |
| Verdict at 2.5 differs, same core (every sample as data) | 22 (2.9%) | 52 (2.9%) |
| of which ringed fails and overlap passes | 21 | 47 |
| Correlation, ringed vs overlap of the stored bitmap | 0.979 | 0.976 |
| Verdict at 2.5 differs, stored bitmap | 22 (2.9%) | 65 (3.6%) |
| Middle square, same core | identical | identical (2 verdicts differ when alpha is read: samples with alpha 0) |

Most of the disagreements are bitmaps the ringed reading sends to `3` through one shift in the outer ring that clears the level by a hundredth or two, such as `(2, 1)` at 0.956 against a level of 0.946; on the overlap the same shift falls just under the level and the radius reads between 1.9 and 2.5. Samples with alpha 0 are rare in consensus bitmaps: 0% on seoul_bull and 0.16% on kerry_park.

## Rust API

The operation lives in [self_similarity/](../../../crates/sfmtool-core/src/patch/self_similarity/) (`mod.rs` for the API, `kernels.rs` for the scalar and AVX2 kernels, `overlap.rs` for the overlap reading, `tests.rs`), as `sfmtool_core::patch::self_similarity`. The bench reads it in [bench/evaluate.rs](../../../crates/sfmtool-core/src/bench/evaluate.rs), the keypoint localizer's member gate in [keypoint_localize.rs](../../../crates/sfmtool-core/src/patch/keypoint_localize.rs) (`member_self_similarity_radius`), cluster-patch refinement's member gate in [cluster_refine/mod.rs](../../../crates/sfmtool-core/src/patch/cluster_refine/mod.rs) (`member_zncc_self_similarity_radius`), and it is bound as `sfmtool._sfmtool.patches.zncc_self_similarity_parts` in [patches/self_similarity.rs](../../../crates/sfmtool-py/src/patches/self_similarity.rs).

```rust
/// The template spread, in grey levels, under which a channel carries no
/// texture.
pub const FLAT_FLOOR: f64 = 0.5;

/// How far a patch can slide over itself and stay indistinguishable from
/// its true position.
pub struct SelfSimilarityParams {
    /// The length of the largest shift searched, in tile pixels; a radius
    /// of `max_radius` reads "this far or further".
    pub max_radius: u32,
    /// ε: the ZNCC deficit two views of the same surface show from warp,
    /// blur and lighting, as a fraction.
    pub relative_tolerance: f64,
    /// n: the noise between two views, in grey levels.
    pub noise: f64,
}

/// One template's reading.
pub struct SelfSimilarity {
    /// How far from the centre the surface falls through `1 − tolerance`,
    /// interpolated linearly along the grid edges, 0 ..= max_radius, in tile
    /// pixels; `max_radius` when an indistinguishable shift lies in the
    /// window's outer ring.
    pub radius: f64,
    /// The direction of the indistinguishable shifts, scaled by how
    /// strongly they line up; `[0, 0]` when there are none.
    pub slide: [f64; 2],
    /// The tolerance τ the template was judged by; infinite for a template
    /// with no textured channel.
    pub tolerance: f64,
    /// z(d) for every shift of the (2r + 1)² square, row-major from
    /// (−r, −r); 1 at the centre, NaN outside the disk.
    pub surface: Vec<f64>,
}

/// A tile's whole core, its middle square and the nine cells of the
/// ZNCC grid's split, each read as its own template.
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

/// One template `Ω = (x, y, w, h)` inside `tile`, with `max_radius`
/// pixels of tile around it on every side.
pub fn zncc_self_similarity_radius(
    tile: &PatchTile<'_>,
    template: [usize; 4],
    params: &SelfSimilarityParams,
) -> SelfSimilarity;

/// The core `R×R` square centred in a `(R + 2r)²` tile, read whole, over
/// its middle `R/4 .. R - R/4` and over each cell of the `R/3`, `R - R/3`
/// split.
pub fn zncc_self_similarity_parts(
    tile: &PatchTile<'_>,
    resolution: usize,
    params: &SelfSimilarityParams,
) -> SelfSimilarityParts;

/// One template inside `tile`, read the overlap way: at each shift, only
/// the template samples whose moved sample lies inside the tile, and where
/// both carry data, are correlated. `data` is one flag per sample, row-major;
/// `None` means every sample carries data.
pub fn zncc_self_similarity_radius_overlap(
    tile: &PatchTile<'_>,
    data: Option<&[bool]>,
    template: [usize; 4],
    params: &SelfSimilarityParams,
) -> SelfSimilarity;

/// A whole `R×R` bitmap with no ring, its middle and its nine cells, each
/// read the overlap way with the rest of the bitmap as its ring.
pub fn zncc_self_similarity_parts_overlap(
    bitmap: &PatchTile<'_>,
    data: Option<&[bool]>,
    params: &SelfSimilarityParams,
) -> SelfSimilarityParts;
```

The ringed functions panic, with a message naming the sizes, on a malformed tile (no channels, more than three, or a value count that does not match), on an empty template, on a template with less than `max_radius` pixels of tile on some side, and, for the parts, on a tile that is not `resolution + 2·max_radius` square or a `resolution` under 3. The overlap functions panic on a malformed tile, on `data` with the wrong number of flags, on an empty template or one that does not fit in the tile, and, for the parts, on a bitmap that is not square or is under `3 × 3`. These are caller errors: the bench and the bindings size the tile from the same parameters they pass.

**Why this shape.** The template is a rectangle inside a larger tile rather than a whole patch because the shifted windows need pixels past the template's edge; a caller that renders the patch renders it `r` pixels wider on every side, and the score never reads outside what it was given. The parts function exists because its eleven templates share one tile and one set of window sums, and computing them together is several times cheaper than eleven calls (see [Implementation notes](#implementation-notes)). The result carries `tolerance` so a display or a test can say why a shift counted, and `surface` because the kernel computes every `z(d)` anyway and a display can draw the surface the radius was read from. The surface is a `Vec` because its size follows `max_radius`. The parameters are a struct and not constants because a caller may need other values than the defaults, and the defaults of `ε` and `n` are not yet calibrated on real pairs of views (see [Open questions](#open-questions)).

```rust
use sfmtool_core::patch::self_similarity::{
    zncc_self_similarity_parts, PatchTile, SelfSimilarityParams,
};

// `bitmap` is a 30 × 30 × 4 render: the 24 × 24 patch and 3 px around it.
let (planes, channels) = PatchTile::planes_from_interleaved(&bitmap, 30, 30, 4);
let tile = PatchTile { values: &planes, channels, width: 30, height: 30 };
let parts = zncc_self_similarity_parts(&tile, 24, &SelfSimilarityParams::default());
if parts.whole.radius < 1.5 {
    // Matching locks onto this patch within one pixel, diagonals included.
}
```

The overlap functions are siblings with an `_overlap` suffix rather than a flag on the ringed ones, because they answer for a different input: a bitmap whose edge is the edge of what exists, which the ringed functions refuse. The suffix names what differs, which is what the shifted windows read, and not the kind of input, since the ringed functions read bitmaps too, rendered wider. They take the data flags as a separate slice rather than reading a fourth channel of the tile, because `PatchTile` holds colour planes only; `data_from_interleaved` derives the flags from an RGBA bitmap in the same layout `planes_from_interleaved` reads.

```rust
use sfmtool_core::patch::self_similarity::{
    zncc_self_similarity_parts_overlap, PatchTile, SelfSimilarityParams,
};

// `bitmap` is a stored 24 × 24 × 4 consensus row: no ring, alpha 0 where no
// view covered the sample.
let (planes, channels) = PatchTile::planes_from_interleaved(&bitmap, 24, 24, 4);
let data = PatchTile::data_from_interleaved(&bitmap, 24, 24, 4);
let tile = PatchTile { values: &planes, channels, width: 24, height: 24 };
let parts = zncc_self_similarity_parts_overlap(&tile, data.as_deref(), &SelfSimilarityParams::default());
let passes = parts.whole.radius <= 2.5; // NaN (no data) is judged by the caller
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

Keypoint lock-in is decided in the first few pixels: a patch that matches itself two or three pixels away will be pulled there by a match that starts nearby, and the exact distance past that does not change what to do with it. A bound of 3 keeps the window at 29 shifts and makes "3" an honest "3 or more". The ZNCC is computed at whole-pixel shifts only, which is enough to see whether the peak is unique, since a ridge or a flat surface shows at whole pixels by construction. The radius is then read between them, where the surface falls through the level, so that two patches whose furthest matching shift is the same whole pixel are told apart by how far past it they keep matching, and a patch that locks reads how sharp its peak is instead of 0.

The radius is a Euclidean length because what decides lock-in is how far from the true position a match could land, in any direction: a shift of `(2, 2)` is 2.83 px away, not 2. A square window would reach 3 along the axes and 4.24 at its corners, so the same edge would score differently turned by 45°. The disk and its saturating outer ring remove that: an edge in any direction that still matches itself at the window's edge scores 3. The disk of radius 1 holds only the four axis shifts, so at `r = 1` an edge at 45° has no shift along itself to match at and scores 0; the bound needs to be at least 2 for every direction to be read.

An exact repeat is also a repeat at every multiple of its step, so a pattern that repeats every `(1, 1)` matches itself at `(2, 2)` as well, which lies in the outer ring at `r = 3` and scores 3. On a pattern that matches itself at `(1, 1)` and stops matching by `(2, 2)`, the radius reads between 1.41 and 2, where the surface falls through the level past `(1, 1)`.

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

The cost is the cross sums, one set per channel. With the template centred, `Σ_k t̃_c(k) (I_c(k + d) − m_c,d) = Σ_k t̃_c(k) I_c(k + d)`, because `t̃_c` sums to zero, so each shift needs one dot product of the template against the moved window in each channel. The window moments `Σ I_c` and `Σ I_c²` over each moved window come from a summed-area table per channel in `f64`, in time proportional to the tile plus the shifts. The channels share the loop structure and are combined only after every shift's per-channel ZNCC is known.

- **Centring and precision.** Each channel of the tile is centred by its own mean over the whole tile and copied into a plane with 8 zero columns of padding per row before either kernel runs, as the localizer's cache is, so the `f32` products stay small and the cross sums keep their precision for a template of a few grey levels. The kernels accumulate raw sums `X(d) = Σ_k I(k) I(k + d)` of the centred plane in `f32` over bands of at most 8 template rows, and each band's sums are added into `f64` totals, which bounds the `f32` rounding a large template accumulates. The centred numerator is then `X(d) − mean_Ω · S1(d)`, with `S1(d)` from the summed-area table. The combine into `z(d)`, the tolerance test and the slide's second moments run in `f64`. The radius is tracked as the largest squared length, an integer, and its one square root is taken after the last shift.
- **Scalar reference.** For each template pixel, each shift row `dy` and each `dx`, accumulate `I(k) · I(k + d)`. It is the oracle for the AVX2 kernel, the fallback on other CPUs and architectures, and the path for `2r + 1 > 8`.
- **AVX2.** Lanes run across the `2r + 1 ≤ 8` horizontal shifts, so one 8-lane register holds a whole shift row of the square that bounds the disk, and `2r + 1` registers hold the window (7 at `r = 3`); the kernel is monomorphized on that count, so the accumulators stay in registers. Per template pixel it broadcasts `I(k)` and, for each shift row, loads the 8 plane values starting at `k + (−r, dy)` and fuses a multiply-add. The lanes past `2r + 1`, and the shifts outside the disk, are computed and discarded: masking them would cost more than the multiply-adds it saves. The padding columns keep every load inside its row's storage, and the bounds that guarantee it are asserted before the kernel runs. It is dispatched at run time on `is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")`, as the localizer's search kernels are ([keypoint-localization-search-cache.md](keypoint-localization-search-cache.md)), and compiled only on `x86_64`. The two kernels agree to `f32` rounding, not bit for bit, since the scalar one rounds each product before adding it.
- **Parts together.** The nine cells tile the core, so the parts function runs the kernel once per cell and sums the cells' `f64` totals into the core's; each part's centred sums then come from its own template mean and the summed-area table. The middle square is not a union of cells and gets its own pass. That is about 35,000 multiply-adds per channel per tile at `R = 24`, `r = 3`, about 105,000 for a colour tile. The AVX2 cross sums take a few microseconds per colour tile, several times less than the scalar ones, and the summed-area tables and the combine over the eleven templates cost more than the cross sums do.
- **The overlap reading.** Its window moments change with the overlap, so a summed-area table of the tile does not give them; for each shift it walks the samples of each part directly and accumulates, per channel, `Σ t`, `Σ u`, `Σ t²`, `Σ u²` and `Σ t u` over the samples where both sides carry data, in `f64`, with each channel first centred by its mean over the samples that carry data. The five sums and the count add, so the nine cells' sums make the whole bitmap's, and the middle gets its own pass. That is about 720 samples per shift at `R = 24`, 29 shifts, and 16 sums per sample for a colour bitmap: roughly 0.4 ms of one core per bitmap, and 12.6 µs per bitmap of wall time for a stack of 17,950 on 32 cores. It has no AVX2 kernel: a mask in the inner loop does not map onto the unmasked cross-sum kernel, and at this cost a batch cull over a whole reconstruction takes well under a second.

## Where it runs

The [keypoint localizer](patch-keypoint-localization.md#the-member-self-similarity-gate) reads the whole core's radius of each view's own tile at its seed offset, from the context tile it has already rendered when that tile has `r` px of ring around the core and from a tile of `R + 2r` rendered for it otherwise, and its member gate, `KeypointLocalizeParams::max_member_zncc_self_similarity_radius`, drops a view whose radius is over the bar. A `NaN` radius fails the gate, `0` turns it off, and the default is `2.5`, chosen by the user from a sweep on seoul_bull and kerry_park ([The member gate's default](patch-keypoint-localization.md#the-member-gates-default)). [Add Image to Tracks](../reconstruction/add-image-to-tracks.md) reads the same radius for the new view's core at the point's projection, reports it per candidate as `zncc_self_similarity_radius`, and refuses a candidate over the same gate as `unlocalizable`.

[Cluster-patch refinement](cluster-patch-refinement.md#which-member-anchors-and-which-members-are-eligible) reads the whole core's radius of each member's own template grid at its SIFT seed geometry, from `sample_member_self_similarity_tile`: the member grid sampled with its half-width grown by `(R + 2r) / R` at resolution `R + 2r`. Its member gate, `ClusterRefineParams::max_member_zncc_self_similarity_radius` (`sfm cluster-patches --max-member-zncc-self-similarity-radius`), refuses a member over the bar as `rejected_unlocalizable` before reference selection, with the localizer's pass rule and its default of `2.5`.

Two batch culls read a point's **consensus bitmap** by the [overlap reading](#the-overlap-reading), through `sfmtool._sfmtool.patches.zncc_self_similarity_parts_overlap_stack` and the shared Python pass rule `points_passing_zncc_self_similarity_radius` in [_filter_by_zncc_self_similarity_radius.py](../../../src/sfmtool/xform/_filter_by_zncc_self_similarity_radius.py): `sfm embed-patches --max-zncc-self-similarity-radius` (`embed_patches(max_zncc_self_similarity_radius=)`) on each point's round-1 consensus, before round 2 ([embed-patches-command.md](../../cli/reconstruction/embed-patches-command.md)), and `sfm xform --filter-by-zncc-self-similarity-radius` (`FilterByZnccSelfSimilarityRadiusTransform`) on a reconstruction's stored `patch_bitmaps` ([xform-command.md](../../cli/reconstruction/xform/xform-command.md)). A point passes at or under the bar and fails on a `NaN` radius; a bitmap with no sample carrying data has no reading and passes, as the bench's painting passes a row with no reading, so a point with no consensus is left to the pipeline's other rules. `0` turns the cull off, `3` or more turns nothing out, and the default is `2.5`, the member gates' `DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS`, which the bindings export under that name. At the default, on the embed-patches runs of the table above, the embed-patches cull drops 59 of 837 points on seoul_bull and 292 of 1,886 on kerry_park; the xform filter over those runs' stored bitmaps, written with the cull off, removes 69 of 765 and 316 of 1,795.

The member gates and the consensus culls ask different questions, and neither makes the other redundant: a point whose members each pin a position can still fuse a consensus that slides, and a point with one textureless member can still fuse a sharp consensus from the others. So the member gates judge each view's own tile before it votes, and the culls judge the fused bitmap.

The `unlocalizable` refusal, the `.matches` member status `rejected_unlocalizable` and the localizer's `N_DROP_UNLOCALIZABLE` counter name what was refused, a view whose position a match cannot pin, rather than the score that judges it. Files written before these gates read the radius carry `max_keypoint_uncertainty` in a `.matches` file's `refine_options` or an `.sfmr`'s `tool_options`; nothing reads those keys back.

The bench computes it at both stages, for every observation that has a pixel (see [editable-track.md](../bench/editable-track.md) § "The ZNCC self-similarity radius"):

- **Track stage.** The tile is the patch rendered through the keypoint-anchored frame, with its half-extent grown by `(R + 2r) / R` and its resolution set to `R + 2r`, so the core is the same `R × R` patch and the ring around it is what the shifted windows read.
- **Cluster stage.** The tile is the one cluster refinement's member gate reads, `sample_member_self_similarity_tile`: the member grid sampled at its seed geometry with the radius grown by the same factor and the resolution set to `R + 2r`. The bench's refinement runs with that gate off, so the bench's whole radius is the number the gate would have judged.

Both read the default `SelfSimilarityParams`. The measurements carry `zncc_self_similarity_radius`, `zncc_self_similarity_radius_middle`, `zncc_self_similarity_radius_grid` (`[[f64; 3]; 3]`), `zncc_self_similarity_slide_grid` (`[[[f64; 2]; 3]; 3]`), `zncc_self_similarity_surface` (the whole core's surface, `Vec<f64>`) and `zncc_self_similarity_tolerance` (the whole core's tolerance, `None` where it is flat), each `None` where the tile could not be rendered or sampled. The bench's `Thresholds::max_zncc_self_similarity_radius` bar judges the whole core's radius, and the threshold painting turns out a row whose radius is over it: a patch that slides over itself that far and still matches, such as a straight edge or a flat patch, does not pin its position. Its default, `BENCH_MAX_ZNCC_SELF_SIMILARITY_RADIUS`, is 2.5 patch-grid px, the same as the localizer's member gate, under the largest shift searched; since the radius reads at most `r`, a bar of `r` or more turns nothing out. A row with no reading clears the bar and a `NaN` fails it. No bar judges the middle, the grid, the slide or the surface.

**Track View** has a *Self-similarity* column, reading the whole radius over the middle one to one decimal (`0.4 px whole` over `1.4 px mid`, with `3+ px` for the maximum) and drawing the grid: green under 1, yellow from 1 to 2, orange from 2 to under `r`, red at `r` or more, with a line along the slide in a cell whose slide is at least 0.5 long, and beside the grid the core's surface plot: the surface interpolated between the shifts and drawn as a heatmap, with the contour at `1 - tolerance` over it and the shifts inside it marked ([track-view.md](../../gui/track-view.md)), and a *self-sim. px* box that sets the bar, from 0 to `r` to one decimal. `get_bench_track` reports the same fields in both blocks and the bar in its `thresholds`, which `apply_bench_track_thresholds` sets, the grids as three rows of three and the surface as rows of numbers with null outside the disk ([mcp-server.md](../../gui/mcp-server.md)), and the Python observation dicts carry them as floats, float64 `(3, 3)` and `(3, 3, 2)` arrays, and a `(2r + 1, 2r + 1)` float64 surface, with the bar in `EditableTrack.thresholds` and as a keyword of `apply_thresholds`.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `max_radius` | `3` | The length of the largest shift searched, in tile pixels; the radius saturates here. |
| `relative_tolerance` | `0.05` | ε, the ZNCC deficit a true match between two views shows from warp, blur and lighting. |
| `noise` | `2.0` | n, the noise between two views, in grey levels. |
| `FLAT_FLOOR` | `0.5` grey levels | A channel whose template spread `s` is under this has no texture; a template with no channel at or above it scores `max_radius`. |

The three parameters' defaults are defined on `SelfSimilarityParams::default()`, and the flat floor is the constant `FLAT_FLOOR` in [self_similarity/mod.rs](../../../crates/sfmtool-core/src/patch/self_similarity/mod.rs).

## Python bindings

`sfmtool._sfmtool.patches.zncc_self_similarity_parts(tile, resolution, *, max_radius=3, relative_tolerance=0.05, noise=2.0)` takes an `(R + 2r, R + 2r)` single-channel tile or an `(R + 2r, R + 2r, C)` patch, uint8 or float32, with a fourth channel read as alpha and dropped, and returns a dict of `radius` (float), `radius_middle` (float), `radius_grid` (`(3, 3)` float64), `slide` (`(2,)` float64), `slide_grid` (`(3, 3, 2)` float64), `tolerance` (float) and `surface` (`(2r + 1, 2r + 1)` float64), the last three of the whole core. A tile of another dtype, rank or size, or with more than four channels, raises `ValueError`. The bench fields arrive through the existing observation dicts.

```python
from sfmtool._sfmtool.patches import zncc_self_similarity_parts

out = zncc_self_similarity_parts(tile_30x30x3_uint8, 24)
out["radius"], out["radius_grid"], out["surface"][3, 3]  # 1.0 at the centre
```

`sfmtool._sfmtool.patches.zncc_self_similarity_parts_overlap_stack(bitmaps, *, max_radius=3, relative_tolerance=0.05, noise=2.0)` reads each bitmap of an `(N, R, R, C)` uint8 or float32 stack by the [overlap reading](#the-overlap-reading), in parallel over the bitmaps, a fourth channel read as the data flags (alpha above 0), and returns a dict of per-bitmap arrays: `radius` (`(N,)` float64), `radius_middle` (`(N,)`), `radius_grid` (`(N, 3, 3)`), `slide` (`(N, 2)`), `tolerance` (`(N,)`), the whole bitmap's where not said otherwise, and `covered` (`(N,)` bool), false for a bitmap with no sample carrying data, whose values are `NaN`. A stack of another dtype or rank, of bitmaps that are not square or under `3 × 3`, or with more than four channels, raises `ValueError`. It is a module function rather than a `PatchCloud` method because it reads the bitmaps alone: no geometry, no views. `DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS` (`2.5`) is exported beside it.

```python
from sfmtool._sfmtool.patches import zncc_self_similarity_parts_overlap_stack

out = zncc_self_similarity_parts_overlap_stack(recon.patch_bitmaps)
passes = ~out["covered"] | (out["radius"] <= 2.5)
```

## Testing

In [self_similarity/tests.rs](../../../crates/sfmtool-core/src/patch/self_similarity/tests.rs):

- The scalar and AVX2 kernels agree on every shift's ZNCC within `1e-4` over random rough and smooth tiles, on the radius within `1e-3` (it is interpolated from the ZNCC, so it carries their rounding, amplified where the values either side of a crossing are close), and on the slide and tolerance exactly, at `r` from 1 to 3, templates from 4 × 4 to 24 × 24, and one to three channels (`avx2_matches_scalar`).
- A red-and-green edge of equal luminance scores as an edge, not as flat; a channel flat in the template is left out; a grey tile repeated in three channels scores as the single channel does.
- A strong corner and a blob score under 1 and more than 0; a straight edge at 0°, 30°, 45° and 90° scores `r` at `r = 2` and `r = 3`, with its slide along the edge; a flat tile, a tile of pure 1-level noise, and an 8-bit sky ramp score `r`; a pattern repeating every 2 px along one axis scores between 2 and 2.3, and one that matches itself at `(1, 1)` but not at `(2, 2)` between 1.41 and 2.
- The radius is where the ZNCC, interpolated linearly along a grid edge, equals the level.
- The surface is 1 at its centre and `NaN` outside the disk.
- The parts function agrees with separate calls on each part's template.
- A tile with too little margin around the template, and a parts tile of the wrong size, are refused with a panic message naming the sizes.
- The overlap reading: a corner locks (under 1); a straight edge at 0°, 30°, 45° and 90° reads 3 with its slide along the edge; a flat bitmap and an 8-bit ramp read 3; samples without data drop out, so what they hold does not change the reading, which equals that of the covered columns cut out on their own, and a cell with no data, or a bitmap with none, has no reading; on a bitmap cut from a larger textured tile, the middle and the centre cell read as the ringed reading of the larger tile does and the whole bitmap within 0.25 of it, with the same tolerance; the parts agree with separate calls; the wrong number of data flags is refused; and `data_from_interleaved` reads alpha.

The bench's track and cluster evaluations fill every field for each observation with a pixel, the painting turns out a row whose radius is over the bar, keeps a row with no reading, and turns out nothing with the bar at the largest radius searched, and the default bar sits under that radius ([bench/tests.rs](../../../crates/sfmtool-core/src/bench/tests.rs)); the viewer's tests cover the column's text, colours, marks and heading ([track_view/edit/tests.rs](../../../crates/sfm-explorer/src/track_view/edit/tests.rs)) and the wire fields ([mcp/tests/bench.rs](../../../crates/sfm-explorer/src/mcp/tests/bench.rs)); and [test_self_similarity_rust_bindings.py](../../../tests/rust_bindings/test_self_similarity_rust_bindings.py) covers the binding. The localizer's tests ([keypoint_localize/tests.rs](../../../crates/sfmtool-core/src/patch/keypoint_localize/tests.rs)) check that its member gate drops a flat and an edge view and keeps textured ones at a bar of 2, drops nothing at `0` or at `r`, gives the same verdicts when the search is too narrow for the ring, passes at or under the bar and fails `NaN`, is on at 2.5 by default, and that a reference search reports the radius; [test_add_image_to_tracks_rust_bindings.py](../../../tests/rust_bindings/test_add_image_to_tracks_rust_bindings.py) checks that a bar refuses as `unlocalizable` exactly the candidates whose radius is over it. Cluster refinement's tests ([cluster_refine/tests.rs](../../../crates/sfmtool-core/src/patch/cluster_refine/tests.rs)) check that a textured member reads well under the bar and a flat, an edge and a smooth member over it, that the tile's core is the member grid, that at the default a flat and an edge member are refused and a textured one kept, that `0` and `3` refuse nobody, and the pass rule. [test_filter_by_zncc_self_similarity_radius.py](../../../tests/xform/test_filter_by_zncc_self_similarity_radius.py) checks the overlap binding on a flat, an edge, a textured and an empty bitmap, the shared pass rule (the default culls the flat and the edge point and keeps the textured and the empty one, `0` and `3` keep every point), and the xform filter on those bitmaps and on real ones; [test_embed_patches_command.py](../../../tests/patch/test_embed_patches_command.py) checks the embed-patches cull on the same four kinds, its default of 2.5, and that its flag reaches `embed_patches` and the written `tool_options`.

## Non-goals

- The overlap reading is not used where a ring can be rendered: the bench and the member gates render the tile `r` px wider and read it the ringed way.
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
