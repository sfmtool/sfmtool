# Context Tile and SIMD Search for Keypoint Localization

Keypoint localization places each view of a 3D point by sliding the view's patch
tile over a small window and scoring, at every shift, how well it matches the
point's reference render; that discrete search, and the render it reads, are
most of the time `sfm embed-patches` spends localizing. This spec covers what
makes them cheap: each view's **context tile** rendered once, wide enough for
every shift the search tries, and hand-written AVX2 kernels that score shifts
from it, either the whole shift grid at once or one cell at a time along a
local descent. The algorithm they serve is specified in
[patch-keypoint-localization.md](patch-keypoint-localization.md); the structure
follows the [fronto-parallel patch cache](fronto-parallel-patch-cache.md):
render once, score many.

Its scope is the **integer** search: putting each keypoint in the right integer
cell, with a quadratic estimate of the sub-pixel step. Getting from there to an
accurate sub-pixel keypoint is a separate algorithm,
[keypoint-subpixel-refinement.md](keypoint-subpixel-refinement.md), which takes
this one's output as its seed.

## Where it lives

The context tile (`ContextTile`, `render_context`, with padded `cache_istride`
rows) is in
[keypoint_localize.rs](../../../crates/sfmtool-core/src/patch/keypoint_localize.rs),
the two searches (`search_shift` for the whole grid, `search_shift_plus_descent`
for the descent) and their reused `SearchScratch` in
[keypoint_localize/search.rs](../../../crates/sfmtool-core/src/patch/keypoint_localize/search.rs),
and the AVX2 kernels (`compute_channel_grids_avx2` for the whole grid,
`score_cell_one_channel_avx2` for one cell) with their scalar fallbacks,
dispatched at runtime, in
[keypoint_localize/kernels.rs](../../../crates/sfmtool-core/src/patch/keypoint_localize/kernels.rs).
Opt-in phase timing (`SFMTOOL_PROFILE=1`) is in
[keypoint_localize/prof.rs](../../../crates/sfmtool-core/src/patch/keypoint_localize/prof.rs).
None of it is public: a caller chooses the search through
`KeypointLocalizeParams::search_strategy` and sizes it through
`KeypointLocalizeParams::search`, and can ask what one view's tile costs with
`view_cache_bytes(params, channels)`.

Sizes at the defaults: `R = 24` (resolution), `search = 6`, `margin =
⌈search⌉ = 6`, the shift grid `span = 2·margin + 1 = 13` on a side (169 cells),
the context tile `R + 2·margin = 36` on a side, the support `n ≈ 452` pixels
(the `GaussianDisk` window over `R²`), and 3 channels (RGB).

## The context tile

For each view other than the reference, the localizer renders one tile,
centred on the view's starting keypoint: the `R×R` core extended by `margin`
grid px on every side, so `R + 2·margin` on a side. The search reads every
candidate shift `(dy, dx)`, `|dy|, |dx| ≤ margin`, from that one tile, and the
member self-similarity gate reads the core at the zero shift from it too.
The reference's own render, the template, is a tile of side `R` with no margin.

**Why reading the tile at an integer shift is exact.** During localization the
patch's centre, axes and normal are fixed; only the in-plane offset varies. A
grid pixel `g` rendered at offset `δ` samples the source at
`project(center + (g + δ)·axes)`, which is exactly the tile's sample at grid
position `g + δ`. So reading the tile at an integer shift is bit-identical to
rendering the patch at that shift, and the search touches the source pyramid
once per view. The exactness holds only for **integer** shifts, which is why
the search reads integer cells and keeps its sub-pixel step as a separate
estimate, a quadratic fitted to the 3×3 cells around the integer peak (a
per-axis parabola where that fit cannot be made, § "Sub-pixel hand-off"),
that never becomes a fractional read.

**Rendering** is the same `WarpMap::from_patch` and remap every patch render
uses, with the sampler chosen once for the view at its starting keypoint. Pixels
map to the patch grid one to one; there is no supersampling.

**Layout.** Planar per channel, in **centered `f32`**:
`plane[c][row·istride + col]` holds `I − c̄_c`, where `c̄_c` is that channel's
mean over the tile. Centering is what makes `f32` accurate (below). Rows are
padded so a 16-wide aligned load from any support column stays in bounds:
`istride = align_up(side − 1 + 16, 8)`; the pad columns hold `0` (the mean after
centering), which only feeds grid cells that are discarded.

**Validity.** A per-pixel invalidity plane (`1.0` out of frame, else `0.0`) is
built with the tile in the same row layout, beside a `bool` validity map. A
shift whose window has any out-of-frame support pixel cannot be scored. The
tile also records whether every pixel is in frame; then the invalidity plane
is all zero (its pad columns too), so both strategies skip counting
out-of-frame pixels, which could only add zeros.

**Memory.** At the defaults one view's tile is about 15 KB and stays in L1
while it is searched. The side grows with `search`, so the tile grows as its
square; the tile planes and the shift grids are reserved fallibly
([patch-keypoint-localization.md](patch-keypoint-localization.md#implementation-details)).

## The search kernel (whole grid, centered f32, register-blocked AVX2)

Windowed ZNCC over the shift grid, per kept channel `c`, as three correlation
maps:

```
Ncross_c(s) = Σ_k kern_c[k]·I_c[s+k]     kern_c[k] = √w[k]·tmpl_c[k]
S1_c(s)     = Σ_k w[k]·I_c[s+k]
S2_c(s)     = Σ_k w[k]·I_c[s+k]²
zncc_c(s)   = (Ncross_c − mean_c·Σ√w·tmpl_c) / √(S2_c − S1_c²/W)   (0 if var < FLAT_EPS)
ZNCC(s)     = (1/channels) Σ_c zncc_c(s)        mean_c = S1_c/W
```

(`s` runs over the `span×span` grid; `W = Σ w`. The template is the reference
render's core, z-normalized with `√w` folded in. The mean term is carried
explicitly, so the result is algebraically identical to z-normalizing each
candidate core and taking its dot product with the template.)

**Register-blocked loop**, in the order channel → grid row `gy` → support pixel
`k`, holding the row's accumulators in registers across the `k` loop so the
tile streams through once and the grids never round-trip to memory in the hot
loop:

```
for c in channels:                         // plane_c (centered), kern_c, w, Σkern_c
  for gy in 0..span:
    n_lo=n_hi=s1_lo=s1_hi=s2_lo=s2_hi = 0  // 6 YMM accumulators (2× __m256 per map)
    for k in 0..n:
      off = gy·istride + off_k             // off_k = r_k·istride + c_k
      src_lo = loadu(plane_c[off..]); src_hi = loadu(plane_c[off+8..])   // 16 cols
      kb = bcast(kern_c[k]); wb = bcast(w[k])
      n_lo  = fma(kb, src_lo, n_lo);  n_hi  = fma(kb, src_hi, n_hi)
      s1_lo = fma(wb, src_lo, s1_lo); s1_hi = fma(wb, src_hi, s1_hi)
      sq_lo = mul(src_lo,src_lo); sq_hi = mul(src_hi,src_hi)
      s2_lo = fma(wb, sq_lo, s2_lo); s2_hi = fma(wb, sq_hi, s2_hi)
    combine this row's `span` cells → zncc_c, add into the combined grid
```

The inner loop is 2 loads, 2 multiplies, 6 FMAs and 2 broadcasts for 16 lanes,
with about 12 YMM registers live. The three maps share each load; padding the
13-cell row to 16 lanes wastes about 19% of them. The combine step (per row,
after the `k` loop) is scalar over `span` cells per channel, which is small.
The per-support-pixel offsets `off_k` are computed once per search, into
`SearchScratch`, rather than by a division and a remainder for every `k` and
`gy`; the address is the same integer, so the result is bit-identical.
Measured with the harness ([scripts/keypoint_localization/](../../../scripts/keypoint_localization/README.md)),
five single-threaded runs alternating with a build without these two changes,
from the stored keypoints: the keypoints and scores of every exhaustive run on
both ground truths are identical, and the time per track falls by 18% on
seoul_bull and 15% on kerry_park; on 40 long DnDTabletop tracks no change was
measurable against the run-to-run spread.
A separate single-accumulator pass over the invalidity plane, with the same
structure and offsets, marks unscorable shifts; it is skipped for a tile with
every pixel in frame. The argmax and the 3×3 quadratic
sub-pixel fit follow.

### Why centering enables f32

The denominator `S2 − S1²/W` cancels catastrophically in `f32` when `I ~ 10²`
(`S2 ~ 10⁷`). Centering the tile by the per-channel mean makes `S1 ≈ 0` and
`S2 ≈ variance · W`, so the cancellation goes away and `f32` is accurate, which
buys the 8-lane width over 4-lane `f64`. The numerator is recovered exactly:
`Ncross = Ncross' + c̄·Σkern`.

## Search strategy: whole grid or "+"-descent

`KeypointLocalizeParams::search_strategy` chooses which cells of the
`(2·margin+1)²` shift grid are scored. Both strategies share the tile, the
support, the template and the result: the integer peak, its sub-pixel step
from the quadratic fitted to the 3×3 cells around it, and the ZNCC at the
integer peak.

The choice is coupled to the kernel:

- **Whole grid → accumulation kernel** (`Exhaustive`, the default). Its cost
  does not depend on the shape of the correlation surface, it is SIMD-friendly,
  and it returns the global maximum over the window, an exact tie going to the
  cell nearer the start (a flat view scores 0 at every shift and stays put).
- **Local descent → per-cell kernel** (`PlusDescent`). It visits few cells but
  pays the full support gather for each, so it wins only while it visits few
  enough, and it returns the peak nearest the starting keypoint rather than the
  highest one in the window.

**Why the whole grid is the default.** On the seoul_bull and kerry_park ground
truths, with every view started 0 to 3 px from its ground-truth keypoint, the
whole grid's keypoints re-triangulate as well as the descent's from within
1 px and better from 2 px, where the descent stops at a nearer, lower peak.
It costs 113 µs per search against the descent's 50 µs at the default window
(13×13 cells), which is 1.1 to 1.2 times the descent's time per track, since
the renders dominate. The measurements are in
[patch-keypoint-localization.md](patch-keypoint-localization.md#how-the-alignment-was-measured).
The accumulation kernel's AVX2 path covers spans up to 16 cells; the span is
`2·⌈search⌉ + 1`, so that is `search` up to 7. A wider window runs the scalar
kernel, and at `search` 9 the whole grid costs 1.2 to 2.0 times the descent
per track (1.7 to 2.0 on DnDTabletop, 1.2 to 1.8 on DinoLedge).

### "+"-descent

Steepest ascent on the integer shift grid: start at `(dy, dx) = (0, 0)`, the
view's starting keypoint, score the 4 axis neighbours, move to the best one that
improves on the current cell, and stop when none does. Each cell is scored at
most once; the visited cache is a dense `Vec<f64>` of the grid's size, with
`NaN` for unvisited, `−∞` for visited and unscorable, and the ZNCC otherwise.
Where the walk stops it scores the 4 diagonal cells around the cell, which the
fit needs for its cross term; if one of them beats the cell, the walk moves to
the best such diagonal and goes on. So the fit is always centred on a cell no
scored neighbour beats, and reuses the 4 axis neighbours already scored to find
that the walk had stopped and the 4 diagonals. A walk that never moves
diagonally visits `9 + 3 · walk_steps` cells: the start and its 4
neighbours, 3 new cells per step, and the 4 diagonals. Neighbours past `±margin` or with any
out-of-frame support pixel are skipped.

**Per-cell scoring (`score_cell_one_channel`).** The AVX2 kernel processes 8
support pixels per iteration with `_mm256_i32gather_ps` over a per-batch index
vector (`(win_y + r_k)·istride + (win_x + c_k)`), with three FMAs per iteration
into 8-lane accumulators for `Σ kern·I`, `Σ w·I` and `Σ w·I²`, then a horizontal
reduction and a scalar tail. The gather is the bottleneck, about 4.1 µs per cell
on dino, but a walk of a few cells still costs a fraction of the whole grid's
accumulation (about 31 µs per search against about 145 µs on dino at the
defaults). The scratch the descent uses (`pd_kerns`, `pd_tsums`,
`pd_per_channel`, `pd_visited`) is reused across every view of a point, so it
allocates nothing after the first view.

## Sub-pixel hand-off

The search's sub-pixel step is the vertex `−H⁻¹g` of the quadratic through the
3×3 cells around the integer peak, with the gradient `g` and Hessian `H` from
central differences, including the cross term
`H_xy = (f(1,1) − f(1,−1) − f(−1,1) + f(−1,−1)) / 4`. The view's final offset is
its starting offset plus the integer peak plus that step. The tile is never read
at a fractional position.

The cross term is why the fit is two-dimensional. On a texture with diagonal
structure the correlation peak's axes are tilted, and a separate parabola per
axis through the integer peak finds the maximum of that row and column, not of
the surface: part of a shift along x shows up on y, by an amount that grows with
the fractional shift and flips sign when the integer peak rounds the other way.
In the unit test `a_fractional_shift_on_one_axis_does_not_show_up_on_the_other`
the per-axis parabola was off by up to 0.40 grid px on a diagonal texture and
0.12 on the test scene's default texture; the 3×3 fit is off by under 0.08 and
0.05. When a diagonal cell could not be scored, or `H` is not negative definite
(the 3×3 cells do not describe a peak), the step falls back to a parabola per
axis, and an axis with an unscored neighbour stays at the integer cell. Each
axis of the step is clamped to one cell. An accurate sub-pixel offset is a
separate, continuous photometric (ECC) solve that optimizes the same template
match with gradients, seeded by the localizer's keypoints:
[keypoint-subpixel-refinement.md](keypoint-subpixel-refinement.md). Keeping it
separate is what lets this search stay integer-exact.

The sub-pixel refiner has a render-once tile of its own (`RefineTile`), on the
same idea, but read at fractional positions through prefiltered cubic B-spline
coefficients and carrying analytic gradient planes, since a continuous solve
cannot restrict itself to integer reads. Its exactness contract is weaker than
this tile's bit-exact integer reads; see "Render-once context tile" in
[keypoint-subpixel-refinement.md](keypoint-subpixel-refinement.md).

## An integer `i16` kernel: investigated, not built

The source is `u8`, so in principle the tile could be `u8`/`i16` and the hot
accumulation could use integer SIMD (`_mm256_madd_epi16` fuses multiply and add
over 16 `i16` lanes; `_mm256_sad_epu8` sums `u8` nearly free), potentially twice
the `f32` lanes. `Ncross` (`Σ kern_q·I`) and `S1` (`Σ I`) convert cleanly. The
obstacle is `S2 = Σ w·I²`: `I²` is 16-bit, the per-pixel weight does not fuse
into `madd`, and an `i32` accumulator can overflow over about 450 pixels. The
clean integer route normalizes with a box window while keeping the Gaussian
weights only in the numerator kernel, which approximates the ZNCC denominator
and is no longer equivalent to the reference.

A prototype (scalar reference plus AVX2 with `vpmulld`/`vpaddd` over `i32`
lanes, a `u8` tile plane and a kernel switch) was benchmarked against the `f32`
kernel on dino (18 961 points) and failed both of its gates:

- **No speedup.** 98.7 µs per search against 93.6 µs for `f32`, about 5% slower:
  `vpmulld` has 5-cycle latency and half the throughput of an FMA, and with 8
  cells per half it has the same lane count as `f32`. The "twice the lanes"
  design needs the harder `_mm256_madd_epi16` horizontal pair-sum layout, which
  was not attempted, since the gather pattern was already about 88% of the
  search's time and the multiplies were not the main cost.
- **14.67% argmax disagreement** with the `f32` kernel (242 472 of 1 653 238
  searches), although the number of kept points was identical (18 961).

What remains of it in the code is the `search_acc` / `search_combine` /
`search_argmax` sub-phase timers in `keypoint_localize::prof`. A future attempt
should start from the `madd_epi16` lane-packing design or address the gather
pattern instead.

## Numerical fidelity and validation

- **The whole-grid kernel against the per-candidate reference.**
  `search_shift` is algebraically identical to extracting, z-normalizing and
  dotting each candidate core (`search_shift_ref`, test-only); the tests check
  the ZNCC grid to a relative tolerance of about `1e-3` and the **exact argmax**
  on clear-peak fixtures (a template cut from the tile's own core at a known
  shift), including a flat channel, an out-of-frame region and a dropped
  channel (`search_shift_matches_reference*`).
- **AVX2 against scalar.** The centered-`f32` scalar forms are the reference and
  the non-x86 / non-AVX2 fallback; `compute_channel_grids_avx2_matches_scalar`
  checks the whole-grid kernel within `f32` tolerance, and
  `score_cell_matches_compute_channel_grids` checks the per-cell kernel against
  the whole-grid kernel's value at every cell of the grid.
- **End to end.** The localizer's own accuracy, against ground truth, is measured
  in [patch-keypoint-localization.md](patch-keypoint-localization.md#how-the-alignment-was-measured);
  the final sub-pixel accuracy is owned by
  [keypoint-subpixel-refinement.md](keypoint-subpixel-refinement.md).

## Design choices

- **One render per view.** The patch frame is fixed during localization, and an
  integer in-plane shift is an integer index shift in the tile, so one render of
  side `R + 2·margin` around the starting keypoint covers every shift the search
  tries, exactly.
- **Centered `f32`.** Removes the variance cancellation that would otherwise
  rule out `f32`, which allows the 8-lane width with no loss of accuracy.
- **Register-blocked on the grid row.** Keeps the whole-grid kernel's hot loop
  in register FMAs with the tile streamed once.
- **Integer-only reads; sub-pixel is a separate algorithm.** Integer reads keep
  every tile access exact; the quadratic estimate is the localizer's sub-pixel
  step, and accuracy is owned by the continuous solve. There is no
  supersampled search grid: it would enlarge the most expensive step for a
  sub-pixel result the continuous solve gives better.

## Open questions

- **The centering constant**: the per-channel tile mean (best conditioning)
  against a fixed `127.5` (cheaper). Whether the fixed constant is accurate
  enough is not measured.
- **The combine step**: scalar against AVX2 `rsqrt`, only worth vectorizing if
  it shows up in a profile.
