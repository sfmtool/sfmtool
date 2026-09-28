# ZNCC Self-Similarity Radius

A keypoint is only worth keeping if matching can find it again: when the patch around it is compared with another photograph, the best match should sit at one place and nothing else nearby. The ZNCC self-similarity radius measures that from a single image. It slides the patch over itself by whole pixels, up to a small maximum distance, and asks how far it can move while still looking as much like itself as a true match between two views would. A corner or a busy texture stops matching itself within a pixel and scores 0 or 1. A straight edge keeps matching itself along its own length and scores the maximum, and so does a flat sky, whose only texture is noise and rounding. Alongside the distance, the score reports the direction the patch can slide in, and the patch's ZNCC with itself at every shift it tried. It is computed for the whole patch, its middle, and each ninth of it, like the bench's other patch readings.

It sits beside the older [patch localizability](patch-localizability.md), the curvature of the same surface at its peak, whose code and bench names carry a `_deprecated` suffix while the two are compared (see [Renamed names of the localizability score](#renamed-names-of-the-localizability-score)). No gate reads the radius.

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

The **radius** is the length `|d| = √(dx² + dy²)` of the furthest indistinguishable shift, or 0 when there are none. It is computed as the largest squared length `dx² + dy²`, a whole number, with one square root at the end. When an indistinguishable shift lies in the disk's outermost ring, `(r − 1)² < dx² + dy² ≤ r²`, the patch still matched itself at the edge of the window and may slide further, so the radius is `r`, read as "`r` or more", whatever that shift's own length. Every direction saturates at the same value this way. At `r = 3` the radius takes the values 0, 1, 1.41, 2, 2.24 and 3. A template with no channel at or above the flat floor has no texture to match and scores `r`, with a slide of `[0, 0]`, an infinite tolerance and a surface of `NaN`.

The **slide** is the direction of the indistinguishable shifts. With `C = Σ d dᵀ / |A|` over the set `A` of indistinguishable shifts and eigenvalues `μ₁ ≥ μ₂`, the slide is the unit eigenvector of `μ₁` scaled by `1 − μ₂ / μ₁`, in the grid frame (`x` column-right, `y` row-down). A set strung out along a line gives a slide near unit length along it, a set spread evenly around the centre gives one near zero, and an empty set gives `[0, 0]`. Its sign means nothing.

The **surface** is `z(d)` itself over the `(2r + 1)²` square of shifts, row-major from `(dx, dy) = (−r, −r)`, with `1` at the centre and `NaN` for the shifts outside the disk. It is what the radius and the slide are read from, returned so a display can draw it.

## Rust API

The operation lives in [self_similarity/](../../../crates/sfmtool-core/src/patch/self_similarity/) (`mod.rs` for the API, `kernels.rs` for the scalar and AVX2 kernels, `tests.rs`), as `sfmtool_core::patch::self_similarity`, a sibling of [`patch::localizability`](../../../crates/sfmtool-core/src/patch/localizability/). The bench reads it in [bench/evaluate.rs](../../../crates/sfmtool-core/src/bench/evaluate.rs), and it is bound as `sfmtool._sfmtool.patches.zncc_self_similarity_parts` in [patches/self_similarity.rs](../../../crates/sfmtool-py/src/patches/self_similarity.rs).

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
    /// The length of the furthest indistinguishable shift, 0 ..= max_radius,
    /// in tile pixels; `max_radius` when one lies in the window's outer ring.
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
```

Both functions panic, with a message naming the sizes, on a malformed tile (no channels, more than three, or a value count that does not match), on an empty template, on a template with less than `max_radius` pixels of tile on some side, and, for the parts, on a tile that is not `resolution + 2·max_radius` square or a `resolution` under 3. These are caller errors: the bench and the binding size the tile from the same parameters they pass.

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

## Theory

### Why a curvature is not enough

The localizability score is the curvature of the same ZNCC surface at `d = 0`, read through the structure tensor: `σ_pos = σ_noise / √λ₂`. It answers "once a match is on the peak, how precisely can the peak be placed under a little independent noise", and it answers that correctly. It does not answer "will a match from a nearby start lock onto this position and no other", for three reasons.

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

Keypoint lock-in is decided in the first few pixels: a patch that matches itself two or three pixels away will be pulled there by a match that starts nearby, and the exact distance past that does not change what to do with it. A bound of 3 keeps the window at 29 shifts, keeps the score to a handful of values a person reads at a glance, and makes "3" an honest "3 or more". Whole-pixel shifts are enough: the question is whether the peak is unique, and a ridge or a flat surface shows at whole pixels by construction.

The radius is a Euclidean length because what decides lock-in is how far from the true position a match could land, in any direction: a shift of `(2, 2)` is 2.83 px away, not 2. A square window would reach 3 along the axes and 4.24 at its corners, so the same edge would score differently turned by 45°. The disk and its saturating outer ring remove that: an edge in any direction that still matches itself at the window's edge scores 3. The disk of radius 1 holds only the four axis shifts, so at `r = 1` an edge at 45° has no shift along itself to match at and scores 0; the bound needs to be at least 2 for every direction to be read.

An exact repeat is also a repeat at every multiple of its step, so a pattern that repeats every `(1, 1)` matches itself at `(2, 2)` as well, which lies in the outer ring at `r = 3` and scores 3. The radius reads 1.41 on a pattern that matches itself at `(1, 1)` and stops matching by `(2, 2)`.

### What the defaults rest on

On 1,350 patches from `seoul_bull_sculpture` and `kerry_park`, drawn evenly from four kinds (flat, faint texture, edge, texture or corner) sorted by measures that none of the scores use, plus crops around seoul_bull's matched keypoints, the radius with `ε = 0.05`, `n = 2` and `r = 3`, measured on a 12 × 12 template of the RGB crop, gives:

| Kind | 0 | 1 to 1.41 | 2 to 2.24 | 3+ |
|---|---|---|---|---|
| flat | 0% | 0% | 0% | 100% |
| faint texture | 16% | 22% | 9% | 53% |
| edge | 37% | 24% | 5% | 35% |
| texture or corner | 92% | 7% | 0% | 1% |
| matched keypoint | 96% | 3% | 0% | 1% |

Scoring luminance instead changes 91 of the 1,350 patches, 80 of them to a longer radius, because it cannot see structure that is in the colour alone. On luminance, without the noise term, a plain 0.95 cutoff scores 51% of the flat patches 0.

## Implementation notes

The cost is the cross sums, one set per channel. With the template centred, `Σ_k t̃_c(k) (I_c(k + d) − m_c,d) = Σ_k t̃_c(k) I_c(k + d)`, because `t̃_c` sums to zero, so each shift needs one dot product of the template against the moved window in each channel. The window moments `Σ I_c` and `Σ I_c²` over each moved window come from a summed-area table per channel in `f64`, in time proportional to the tile plus the shifts. The channels share the loop structure and are combined only after every shift's per-channel ZNCC is known.

- **Centring and precision.** Each channel of the tile is centred by its own mean over the whole tile and copied into a plane with 8 zero columns of padding per row before either kernel runs, as the localizer's cache is, so the `f32` products stay small and the cross sums keep their precision for a template of a few grey levels. The kernels accumulate raw sums `X(d) = Σ_k I(k) I(k + d)` of the centred plane in `f32` over bands of at most 8 template rows, and each band's sums are added into `f64` totals, which bounds the `f32` rounding a large template accumulates. The centred numerator is then `X(d) − mean_Ω · S1(d)`, with `S1(d)` from the summed-area table. The combine into `z(d)`, the tolerance test and the slide's second moments run in `f64`. The radius is tracked as the largest squared length, an integer, and its one square root is taken after the last shift.
- **Scalar reference.** For each template pixel, each shift row `dy` and each `dx`, accumulate `I(k) · I(k + d)`. It is the oracle for the AVX2 kernel, the fallback on other CPUs and architectures, and the path for `2r + 1 > 8`.
- **AVX2.** Lanes run across the `2r + 1 ≤ 8` horizontal shifts, so one 8-lane register holds a whole shift row of the square that bounds the disk, and `2r + 1` registers hold the window (7 at `r = 3`); the kernel is monomorphized on that count, so the accumulators stay in registers. Per template pixel it broadcasts `I(k)` and, for each shift row, loads the 8 plane values starting at `k + (−r, dy)` and fuses a multiply-add. The lanes past `2r + 1`, and the shifts outside the disk, are computed and discarded: masking them would cost more than the multiply-adds it saves. The padding columns keep every load inside its row's storage, and the bounds that guarantee it are asserted before the kernel runs. It is dispatched at run time on `is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")`, as the localizer's search kernels are ([keypoint-localization-search-cache.md](keypoint-localization-search-cache.md)), and compiled only on `x86_64`. The two kernels agree to `f32` rounding, not bit for bit, since the scalar one rounds each product before adding it.
- **Parts together.** The nine cells tile the core, so the parts function runs the kernel once per cell and sums the cells' `f64` totals into the core's; each part's centred sums then come from its own template mean and the summed-area table. The middle square is not a union of cells and gets its own pass. That is about 35,000 multiply-adds per channel per tile at `R = 24`, `r = 3`, about 105,000 for a colour tile. The AVX2 cross sums take a few microseconds per colour tile, several times less than the scalar ones, and the summed-area tables and the combine over the eleven templates cost more than the cross sums do.

## Where it runs

The bench computes it beside the deprecated localizability at both stages, over the same frame, for every observation that has a pixel (see [editable-track.md](../bench/editable-track.md) § "The ZNCC self-similarity radius"):

- **Track stage.** The tile is the patch rendered through the keypoint-anchored frame, with its half-extent grown by `(R + 2r) / R` and its resolution set to `R + 2r`, so the core is the same `R × R` patch and the ring around it is what the shifted windows read.
- **Cluster stage.** The member grid is sampled at its seed geometry with the radius grown by the same factor and the resolution set to `R + 2r`.

Both read the default `SelfSimilarityParams`. The measurements carry `zncc_self_similarity_radius`, `zncc_self_similarity_radius_middle`, `zncc_self_similarity_radius_grid` (`[[f64; 3]; 3]`), `zncc_self_similarity_slide_grid` (`[[[f64; 2]; 3]; 3]`), `zncc_self_similarity_surface` (the whole core's surface, `Vec<f64>`) and `zncc_self_similarity_tolerance` (the whole core's tolerance, `None` where it is flat), each `None` where the tile could not be rendered or sampled, and set and cleared wherever the deprecated localizability is. No bar judges them.

**Track View** has a *Self-sim.* column beside σ_pos, reading the whole and middle radii to two decimals with trailing zeros dropped (`0 / 1.41`, with `3+` for the maximum) and drawing the grid: green at 0, yellow at 1 to 1.41, orange at 2 to 2.24, red at `r` or more, with a line along the slide in a cell whose slide is at least 0.5 long, and beside the grid the core's surface plot: the surface interpolated between the shifts and drawn as a heatmap, with the contour at `1 - tolerance` over it and the shifts inside it marked ([track-view.md](../../gui/track-view.md)). The σ_pos column stays while the two are compared, its heading's hover text saying it is the deprecated score. `get_bench_track` reports the same fields in both blocks, the grids as three rows of three and the surface as rows of numbers with null outside the disk ([mcp-server.md](../../gui/mcp-server.md)), and the Python observation dicts carry them as floats, float64 `(3, 3)` and `(3, 3, 2)` arrays, and a `(2r + 1, 2r + 1)` float64 surface.

## Renamed names of the localizability score

While the two are compared, the localizability score's names say it is the one on its way out. What is renamed is what names the score:

| Name | Was |
|---|---|
| `patch_localizability_deprecated` | `patch_localizability` |
| `score_localizability_stack_deprecated` | `score_localizability_stack` |
| `score_localizability_parts_deprecated` | `score_localizability_parts` |
| `LocalizabilityDeprecated`, `LocalizabilityPartsDeprecated` | `Localizability`, `LocalizabilityParts` |
| `PatchCloud.score_localizability_deprecated` (Python) | `PatchCloud.score_localizability` |
| bench `localizability_deprecated`, `localizability_middle_deprecated`, `localizability_grid_deprecated`, `localizability_slide_deprecated` (Rust, MCP and Python) | the same names without `_deprecated` |

What names a gate or appears in a file or on a command line keeps its name, because renaming it would change a file format or a command-line interface for users of the batch pipeline: the `max_keypoint_uncertainty` and `max_member_keypoint_uncertainty` parameters and flags, the bench's `max_keypoint_uncertainty` bar, `--filter-by-keypoint-uncertainty`, `SIGMA_NOISE`, and the `.matches` member status `rejected_unlocalizable`. Those gates call the deprecated scorer. The module stays `patch::localizability`, and [patch-localizability.md](patch-localizability.md) names the deprecated identifiers and links this spec.

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

## Testing

In [self_similarity/tests.rs](../../../crates/sfmtool-core/src/patch/self_similarity/tests.rs):

- The scalar and AVX2 kernels agree on every shift's ZNCC within `1e-4` over random rough and smooth tiles, and on the radius, slide and tolerance exactly, at `r` from 1 to 3, templates from 4 × 4 to 24 × 24, and one to three channels (`avx2_matches_scalar`).
- A red-and-green edge of equal luminance scores as an edge, not as flat; a channel flat in the template is left out; a grey tile repeated in three channels scores as the single channel does.
- A strong corner and a blob score 0; a straight edge at 0°, 30°, 45° and 90° scores `r` at `r = 2` and `r = 3`, with its slide along the edge; a flat tile, a tile of pure 1-level noise, and an 8-bit sky ramp score `r`; a pattern repeating every 2 px along one axis scores 2, and one that matches itself at `(1, 1)` but not at `(2, 2)` scores 1.41.
- The surface is 1 at its centre and `NaN` outside the disk.
- The parts function agrees with separate calls on each part's template.
- A tile with too little margin around the template, and a parts tile of the wrong size, are refused with a panic message naming the sizes.

The bench's track and cluster evaluations fill every new field wherever `localizability_deprecated` is filled ([bench/tests.rs](../../../crates/sfmtool-core/src/bench/tests.rs)); the viewer's tests cover the column's text, colours, marks and heading ([track_view/edit/tests.rs](../../../crates/sfm-explorer/src/track_view/edit/tests.rs)) and the wire fields ([mcp/tests/bench.rs](../../../crates/sfm-explorer/src/mcp/tests/bench.rs)); and [test_self_similarity_rust_bindings.py](../../../tests/rust_bindings/test_self_similarity_rust_bindings.py) covers the binding.

## Non-goals

- The radius does not replace the deprecated score in any kernel gate or in the `xform` filter; those keep `σ_pos`.
- It does not measure sub-pixel precision, which the deprecated score and the sub-pixel refinement already cover.
- It does not change any file format.

## Open questions

- **Calibrating the defaults of ε and n on real pairs of views.** `relative_tolerance` and `noise` are parameters; their defaults in `SelfSimilarityParams::default()` rest on the single-image sample above and are not checked against matching between two views. The check that would set them starts each sighting's search on the ground-truth tracks of `seoul_bull_sculpture` and `kerry_park` 1, 2 and 3 px off its true keypoint along each axis and along the diagonals, and records whether the localizer's ZNCC search walks back. A sighting's measured lock-in is the largest offset it recovers from, and the radius should predict it: a sighting scored under 1.5 recovers from every offset up to 3, and one scored 3 fails from some. The same test scores the deprecated localizability, and the comparison decides the default tolerance and whether the kernels' gates change. The residuals between views that the track stage already computes are a direct measurement of `n` for it.
- **Sharing the localizer's search kernel.** `keypoint_localize::kernels::compute_channel_grids` already computes cross sums and window moments over a dense shift grid for a weighted template. A self-similarity call is that kernel with the tile's own core as the template; sharing it would mean lifting it out of `keypoint_localize` and giving it an unweighted form.
- **A bar.** Whether the bench paints on the radius, with a `max_zncc_self_similarity_radius` bar, once the calibration settles its meaning.
