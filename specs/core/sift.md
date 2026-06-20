# SIFT

> **Status: Implemented (Phase 1 — CPU + SIMD + multithread).** The pure-Rust
> SIFT detector and descriptor ship in `crates/sfmtool-core/src/sift/`, with
> PyO3 bindings (`py_sift.rs`, `py_sift_io.rs`) and the `sfmtool` backend of
> `sfm sift` / `ws init --feature-tool sfmtool`. Structured to mirror the
> optical-flow implementation (`specs/core/optical-flow.md`). GPU remains
> deferred to a later phase (a future `specs/core/gpu-sift.md`); the on-disk
> incremental-extraction extensions below are likewise still future work.

## Motivation

sfmtool relies on COLMAP and OpenCV for many of its algorithms; one of those is SIFT
feature extraction (`src/sfmtool/sift/extract_colmap.py` and `extract_opencv.py`, both
wrapping external binaries). Adding a Rust implementation directly in sfmtool-core — like
the native optical-flow implementation — would give us more room for flexibility:

- Use in the Rust-only GUI for interactive feature inspection
- Control over the algorithm for SfM-specific tuning (e.g. custom contrast/edge
  thresholds per dataset, deterministic ordering, exact subpixel conventions)
- Integration with the rayon-parallel matching pipeline and `sift-format` I/O
  without a Python/OpenCV round-trip
- A path to GPU acceleration later, reusing the wgpu infrastructure built for
  optical flow

This spec defines library functions in sfmtool-core, independent of any on-disk layout.
Their interface follows COLMAP's conventions rather than OpenCV's: keypoint coordinates use
COLMAP's pixel-center convention (the upper-left pixel's center is `(0.5, 0.5)`), and each
keypoint's geometry is a 2×2 affine shape matrix `[[a11, a12], [a21, a22]]`, exactly as
COLMAP represents features. SIFT produces similarity-only keypoints (location, scale,
orientation), which map onto that matrix as a scaled rotation
(`[[s·cosθ, −s·sinθ], [s·sinθ, s·cosθ]]`).

## Algorithm: Scale-Invariant Feature Transform

Reference: David G. Lowe, "Distinctive Image Features from Scale-Invariant
Keypoints," IJCV 60(2):91–110, 2004.
[ijcv04](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)

### Overview

SIFT is a cascade-filtering detector followed by a gradient-histogram descriptor.
Five stages:

1. **Scale-space (Gaussian) pyramid** — repeated Gaussian blur across `s+3` levels
   per octave, octaves separated by 2x downsampling.
2. **Difference-of-Gaussians (DoG)** — subtract adjacent Gaussian levels; an
   efficient approximation to the scale-normalized Laplacian σ²∇²G.
3. **Scale-space extrema detection** — each DoG sample compared to its 26 neighbors
   (8 in-scale + 9 above + 9 below); keep strict maxima/minima.
4. **Keypoint localization (size determination)** — sub-pixel/sub-scale fit via a 3D
   quadratic, then reject low-contrast and edge-like responses. This is where each
   keypoint's *scale* (size) and refined (x, y) are pinned down.
5. **Orientation assignment + descriptor** — dominant gradient orientation(s) from a
   36-bin histogram, then a 4×4×8 = 128-D gradient-orientation descriptor, normalized
   and clamped.

### 0. Image-to-gray conversion

The detector operates on a single-channel float image `I`, and several parameters live in
that value domain — notably the contrast threshold (`|D(x̂)| < 0.03`, assuming pixels in
`[0, 1]`). The mapping from the source image to `I` therefore changes both the features and
the meaning of those thresholds, so it must be pinned, not left to whatever the decoder
happens to do. It is recorded in the `.sift` metadata as a formula over the normalized
colour channels — `feature_options.image_to_gray = { "formula": "0.2126*R + 0.7152*G +
0.0722*B" }` by default, matching COLMAP's `Bitmap::CloneAsGrey` (so it participates in
`feature_tool_xxh128`). See
[`../formats/sift-file-format.md`](../formats/sift-file-format.md) §"Image-to-gray
conversion" for the formula grammar and channel definitions.

### 1. Scale-space pyramid

Scale space is `L(x, y, σ) = G(x, y, σ) ∗ I(x, y)` with the 2D Gaussian
`G(x, y, σ) = 1/(2πσ²) · exp(−(x²+y²)/2σ²)`.

Each octave is divided into `s` intervals with multiplicative scale step
`k = 2^(1/s)`. **`s = 3`** (Lowe's experimentally-optimal value, Fig. 3). To cover a
*complete* octave during extrema detection (which needs one DoG above and below each
candidate level), we produce **`s + 3` Gaussian images per octave** → **`s + 2` DoG
images** → `s` levels at which extrema are searched.

- **Base scale σ = 1.6** (Fig. 4 — near-optimal repeatability vs. cost).
- **Image doubling.** The input is upsampled 2× with bilinear interpolation before
  the first octave (increases stable keypoints ~4×). The original is assumed to carry
  blur ≥ 0.5 (anti-aliasing minimum); after doubling that is σ = 1.0 relative to the
  new pixel spacing. The first level is then brought to σ = 1.6 with an incremental
  blur of `sqrt(1.6² − 1.0²)`.
- **Incremental blur.** Within an octave, level `i` is produced from level `i−1` by a
  Gaussian of `σ_inc(i) = σ · sqrt(k^(2i) − k^(2(i−1)))`, so the absolute blur of
  level `i` is `σ · k^i`. Incremental blur is cheaper and numerically equivalent.
- **Octave downsampling.** After an octave is built, the Gaussian image with twice the
  base σ (the one `s` levels up, i.e. 2 from the top of the stack) is resampled by
  taking every second pixel in x and y to seed the next octave — same sampling accuracy
  relative to σ, far less computation.
- **Octave count** = `floor(log2(min(W, H))) − offset` (stop when the smallest octave
  is a handful of pixels). The implementation uses `offset = 3`, clamps the count to at
  least 1, and falls back to a single octave when the smallest side is `< 8` px.

Gaussian convolution is **separable** (1D horizontal pass then 1D vertical pass), which
the existing pyramid module already exploits for optical flow (6-tap kernel). For SIFT
the kernel radius scales with σ_inc, so kernels are computed once per blur. The radius is
`ceil(blur_radius_factor · σ)` (`2·radius + 1` taps); the truncated kernel is renormalized
to sum 1. The default `blur_radius_factor` is **2.25** (~97.6% of the Gaussian mass) —
chosen empirically as the narrowest setting that leaves descriptor agreement with COLMAP
intact while cutting taps on the widest (high-σ) blurs, which dominate scale-space cost.
Lowe's wider 3·σ (~99.7% mass) is available via the parameter.

The interior of both passes is vectorized (AVX2+FMA where available, SSE2 otherwise);
`SFMTOOL_SIFT_NO_AVX2` forces the SSE2 path for A-B timing. The horizontal pass is
contiguous read/write; the vertical pass reads a strided column neighborhood but writes
contiguous output rows, and rayon's per-row split gives each thread a sliding band of
`2·radius + 1` source rows that stays resident in L2 (each source row is read from DRAM
once and reused across every output row it contributes to).

**Rejected: transpose-fused passes.** A transpose-fused variant — each pass convolving
along the contiguous axis and writing its result transposed (in-register 8×8 SIMD
transpose), so the vertical pass also becomes a contiguous convolution — was implemented
and benchmarked. It was **~2× slower** (blur 184 vs 94 ms/img on dino_dog_toy) and
produced bit-identical results. The strided vertical read is not the bottleneck (the
sliding-band L2 reuse above already covers it), and the transpose replaces *both* passes'
contiguous writes with scattered transposed writes (eight 32-byte chunks `in_h` apart per
tile), whose partial-cache-line / TLB cost dominates. Do not re-attempt without a
fundamentally cheaper transpose.

### 2. Difference-of-Gaussians

`D(x, y, σ) = L(x, y, kσ) − L(x, y, σ)` — a plain per-pixel subtraction of adjacent
Gaussian levels (eq. 1). Cheap, and a close approximation to σ²∇²G since
`G(x,y,kσ) − G(x,y,σ) ≈ (k−1)σ²∇²G` and the `(k−1)` factor is constant across scale so
it doesn't move extrema.

### 3. Scale-space extrema detection

Each interior DoG sample is compared against its 26 neighbors (3×3×3 minus itself,
spanning the level below, the level itself, and the level above). Selected iff it is
strictly greater than *all* 26 or strictly less than *all* 26. Most candidates die in
the first few comparisons. A cheap pre-threshold (|D| above a fraction of the contrast
threshold) skips obvious flats before the full 26-way test.

### 4. Keypoint localization (size determination)

For each raw extremum, fit a 3D quadratic (Taylor expansion, eq. 2) to the DoG around
the sample point:

    D(x) = D + (∂D/∂x)ᵀ x + ½ xᵀ (∂²D/∂x²) x,   x = (x, y, σ)

The sub-pixel/sub-scale offset is the stationary point (eq. 3):

    x̂ = −(∂²D/∂x²)⁻¹ (∂D/∂x)

The 3×3 gradient and Hessian are estimated by central finite differences over the DoG
neighborhood. If any component of `x̂` exceeds 0.5, recenter on the neighbor and refit
(cap iterations, e.g. 5, then discard). The offset is added to the integer location to
give the refined (x, y) and the keypoint's continuous scale `σ_kp = σ · k^(level + x̂_σ)`
in the octave's coordinates — **this is the "size":** OpenCV reports it as keypoint
diameter `2 · σ_kp`, which maps to the affine-shape magnitude in `.sift`.

Two rejections:

- **Low contrast.** `|D(x̂)| = |D + ½ (∂D/∂x)ᵀ x̂| < C` → discard (pixels in [0,1]).
  Lowe's paper uses `C = 0.03`, but our `|D(x̂)|` is on the same scale as the
  *effective* per-layer threshold the references ship — OpenCV's
  `contrastThreshold / nOctaveLayers` (≈ 0.0133) and COLMAP's `peak_threshold`
  (≈ 0.0067) — so the default is `C = 0.0067` to match COLMAP's keypoint density
  (Lowe's literal 0.03 yields ~7× fewer keypoints than OpenCV).
- **Edge response.** From the 2×2 spatial Hessian `H = [[Dxx, Dxy], [Dxy, Dyy]]`,
  discard if `Det(H) ≤ 0` or `Tr(H)²/Det(H) ≥ (r+1)²/r` with **`r = 10`** — i.e. the
  ratio of principal curvatures exceeds 10 (poorly-localized along an edge).

### 5. Orientation assignment

At the Gaussian level nearest the keypoint's scale, precompute gradient magnitude and
orientation:

    m(x,y) = sqrt((L(x+1,y) − L(x−1,y))² + (L(x,y+1) − L(x,y−1))²)
    θ(x,y) = atan2(L(x,y+1) − L(x,y−1),  L(x+1,y) − L(x−1,y))

Build a **36-bin** orientation histogram over a circular window around the keypoint,
each sample weighted by `m` and by a Gaussian with **σ_w = 1.5 · σ_kp** (window radius
~3·σ_w). Smooth the histogram, take the max peak, and emit a keypoint for the peak and
for every other local peak within **80%** of the max (≈15% of keypoints get multiple
orientations — they materially help matching). Fit a parabola to the 3 bins around each
peak for sub-bin angular accuracy.

### 6. Descriptor

In a window rotated to the keypoint orientation, sample gradients at the keypoint's
Gaussian level and accumulate into a **4×4 array of 8-bin orientation histograms = 128-D**
vector. The 4×4 is the histogram-bin grid, not a fixed sample lattice: every integer
pixel inside the rotated window is sampled and trilinearly distributed (COLMAP-style),
rather than a fixed 4×4 samples-per-subregion grid.

- Each subregion spans `hist_width = m_descr · σ_kp` octave-pixels (Lowe uses
  `m_descr = 3`). To cover the rotated `d × d` grid plus a one-subregion interpolation
  margin, iterate the integer pixels inside the axis-aligned bounding box of half-width
  `radius = round(hist_width · (d + 1) / 2 · sqrt(2))` (with `d = 4`).
- Each gradient is weighted by magnitude and a Gaussian with **σ = half the descriptor
  window width = `d/2` subregion units** (`exp(−r² / (2·(d/2)²))`, distance in subregion units).
- **Trilinear interpolation** spreads each sample across the 2 nearest spatial bins in
  each axis and the 2 nearest orientation bins (weight `1 − d` per dimension).
- Normalize to unit length, **clamp each component to ≤ 0.2**, renormalize. This caps
  the influence of large gradients (non-linear illumination robustness). Finally
  quantize to `u8` (×512, clamp 255) for the `.sift` format, matching OpenCV/COLMAP.

### 7. Feature-count cap

`max_num_features` bounds the number of features per image (COLMAP's
`max_num_features`, default **8192**; `None` disables the cap). When detection yields
more candidates than the cap, the **largest-scale** ones are kept — large features are
the most stable, and this matches COLMAP's selection. The cap is applied to the localized
candidates *before* orientation and description, so those (atan2-heavy) stages only run on
the retained set; a final truncation after orientation enforces the cap as a hard output
limit (orientation can emit several keypoints per candidate). On images that detect fewer
than `max_num_features` candidates the cap is a no-op.

### Parameters

| Symbol | Name | Description | Default |
|--------|------|-------------|---------|
| s | Octave layers | Intervals per octave (Gaussian levels = s+3) | 3 |
| σ | Base sigma | Blur of the first level of octave 0 | 1.6 |
| σ_in | Input blur | Assumed blur of the source image | 0.5 |
| — | Upsample | Double input before octave 0 | true |
| C | Contrast threshold | Discard if \|D(x̂)\| < C (matches COLMAP `peak_threshold`) | 0.0067 |
| r | Edge threshold | Max ratio of principal curvatures | 10 |
| N_max | Max features | Cap on output, keeping largest-scale (COLMAP `max_num_features`); `None` = unlimited | 8192 |
| n_ori | Orientation bins | Bins in the orientation histogram | 36 |
| — | Peak ratio | Secondary-orientation fraction of max peak | 0.8 |
| d | Descriptor width | d×d array of histograms | 4 |
| b | Descriptor bins | Orientation bins per histogram | 8 |
| m_descr | Descriptor magnification | Sample spacing in units of σ_kp | 3 |
| — | Clamp | Descriptor component cap before renorm | 0.2 |

## Parallelism & SIMD strategy

A two-tier SIMD strategy plus **rayon** data parallelism. Hot loops have an
explicit **AVX2 + FMA** path (8-wide f32, selected at runtime via
`sift::simd::HAS_AVX2_FMA`) with an **SSE2** (4-wide) or scalar fallback for
borders, non-x86, and CPUs lacking AVX2/FMA; `SFMTOOL_SIFT_NO_AVX2` forces the
fallback (for A-B timing and reproducibility). Per step:

| Step | Multithreading (rayon) | SIMD |
|------|------------------------|------|
| **Gaussian pyramid** | Octaves are sequential (each seeds the next), but within an octave the separable blur parallelizes by row (horizontal pass) / column-block (vertical pass) via `par_chunks_mut`. Multiple images (image-A/image-B, or a batch of input images) build in parallel with `par_iter`. | Separable convolution is the classic SIMD kernel: the interior loops run 8-wide AVX2+FMA (load contiguous pixels, FMA against each tap) with a 4-wide SSE2 fallback, reusing the optical-flow pyramid's vectorized blur. |
| **DoG** | Not a standalone stage: fused into detection (see *tiled DoG/detect fusion* below). DoG values are computed per row-stripe in cache and never materialized to RAM. | Pure 4-wide subtraction `_mm_sub_ps`. |
| **Extrema detection** | `par_iter` over `(octave, row-stripe)`; each stripe computes its DoG band in a cache-resident scratch buffer and writes its own candidate list (`flat_map`/`collect`). Replaces the coarser per-`(octave, level)` jobs. | The 26-neighbor test is branchy and scalar-friendly, but the *pre-threshold* flat-rejection (`|D| > frac·C`) vectorizes 4 pixels at a time to skip dead regions cheaply. |
| **Localization** | `par_iter` over the candidate list — each candidate is independent; output filtered keypoints via `filter_map().collect()`. | Low arithmetic per candidate (3×3 solve); leave scalar. The win is thread-parallelism over many candidates. |
| **Orientation** | `par_iter` over localized keypoints (each may emit 1–N oriented keypoints → `flat_map`). | Implemented (`sift::simd`): the per-pixel window math runs **8-wide AVX2+FMA** — gradient diff, `sqrt`, a polynomial `atan2_approx` (≈1.3e-3 rad) and the `exp_approx` Gaussian weight — with only the `hist[bin] += …` scatter left scalar. ~2.1× over the scalar fill on dino (46→22 ms/img); SSE2/non-x86 fall back to scalar. |
| **Descriptor** | `par_iter` over oriented keypoints; each fills its own 128-D vector independently. | Implemented (`sift::simd`): each window row runs **8-wide AVX2+FMA** — the rotation into the descriptor frame, gradient diff, `sqrt`, `atan2_approx`, the `exp_approx` Gaussian weight and the orientation-bin reduction — with only the per-sample trilinear scatter left scalar. ~6.5× over the scalar fill on dino (211→33 ms/img). Final L2 norm / clamp / renorm is 4-wide SSE2 (`l2_norm_sse2`/`scale_sse2`); SSE2/non-x86 fall back to scalar. |

Determinism note: rayon's `collect` preserves input order, so per-image keypoint
ordering stays stable across runs (important for reproducible `.sift` output and tests).

The dominant cost is the pyramid (memory-bound, SIMD-bound) and the descriptor
(many independent keypoints, thread-bound) — both map cleanly onto the existing
patterns.

### Scale-space memory traffic: tiled DoG/detect fusion

**Motivation (measured).** Profiling `extract_sift` on dino (2040×1536, so a
4080×3072 octave-0 after image doubling) with `SFMTOOL_SIFT_TIMING` at 1/2/4 threads
shows two stages that *do not* scale — they cap at ~2.6× on 4 cores while blur,
base and orientation reach 3.2–3.5×:

| stage | 1T | 4T | speedup | limiter |
|------|----|----|---------|---------|
| DoG `difference` | 172 ms | 67 ms | 2.6× | memory bandwidth |
| extrema `detect` | 177 ms | 69 ms | 2.6× | memory bandwidth + coarse jobs |

The cause is data movement, not arithmetic. An octave-0 level is 4080×3072×f32 =
**50 MB**. The current pipeline (a) **materializes the whole DoG pyramid** to RAM
(`s+2` levels ≈ 250 MB written for octave 0), then (b) `detect` **reads it back**
(each level serves as `below`/`cur`/`above` for adjacent levels, so ~3 reads), all
streamed from main memory. On a bandwidth-limited host that round-trip — not the
subtraction or the 26-neighbour test — is the wall. A secondary issue: detection
parallelizes over `(octave, level)`, so octave 0 has only `s = 3` jobs — too coarse
to balance 4+ cores.

The fix is to stop touching RAM for the DoG and to expose finer parallelism, in
two tiers. Both rest on one identity: **DoG is pointwise** — `dog(o,l) = g(o,l+1)
− g(o,l)` — and the Gaussian pyramid is *already* resident (orientation and
description sample it after detection). So any DoG value can be recomputed from
two gaussians with **bit-identical** f32 arithmetic (`a − b` is one IEEE
subtraction whether done by `subps`, `subss`, or scalar `-`), and the DoG pyramid
need never exist as a stored array.

#### Tier 1 — fuse DoG into detection, tiled in row stripes (implemented)

Stop materializing `Octave::dogs` entirely. `detect_and_localize` parallelizes
over **`(octave, row-stripe)`** instead of `(octave, level)`. Each stripe task,
for its band of `STRIPE_ROWS` owned interior rows (plus a **1-row halo** for the
3×3×3 test):

1. computes the `s+2` DoG levels **for that stripe only** into a small per-task
   scratch buffer (`(s+2) × (STRIPE_ROWS+2) × W × f32`, a few MB — fits L2/L3),
   from the resident gaussian stripes;
2. runs the existing pre-threshold (`candidate_cols`) + 26-neighbour
   `is_extremum` scan over the owned rows of each interior level `l = 1..=s`,
   reusing the stripe's DoG scratch;
3. for each surviving candidate, runs `localize` — which reads DoG **on the fly**
   from the resident gaussians (`g(o,l+1)[i] − g(o,l)[i]`), giving it the full
   random, multi-level, iterative-recenter access it needs with no halo limit
   (localization is sparse, so this costs nothing in bandwidth).

The DoG of octave 0 thus lives only in cache: the 250 MB write and the read-back
disappear, leaving the gaussians (streamed once per stripe, reused across levels
in-cache) as the only octave-0 detect traffic. Tiling is **safe by construction**:
the DoG scan has a 1-row halo (pointwise DoG, 3×3×3 detect), each stripe *owns* a
disjoint row band and only *reads* the halo, so the keypoint **set** is identical;
the existing total-order sort in `detect_keypoints` then yields **byte-identical
`.sift`** regardless of stripe boundaries or thread count. Bonus wins: peak RSS
drops by the DoG pyramid (~250 MB octave-0, more across octaves), and octave 0
goes from 3 detect jobs to dozens (`rows / STRIPE_ROWS`), fixing the imbalance.
`STRIPE_ROWS` is the one tuning knob (cache-fit vs. negligible halo overhead),
overridable via `SFMTOOL_SIFT_STRIPE_ROWS` and clamped to `[1, 2^16]` so a stray
value can't overflow the stripe loop or collapse an octave into one stripe.

What Tier 1 does **not** touch: the blur chain still materializes the gaussians to
RAM (they must persist for orientation/description), so `base`/`blur`/`decimate`
are unchanged.

#### Tier 2 — fuse the blur chain into the stripe (future)

Keep the gaussians cache-resident too, fusing blur → DoG → detect per stripe so a
stripe is read from RAM once (octave base) and only keypoints come out. Two
obstacles make this a separate, larger effort with a narrower margin:

- **The gaussian pyramid must persist.** Orientation/description sample scattered
  gaussian patches at keypoint locations *after* detect, so either the gaussians
  are still written to RAM (no blur-bandwidth saved) or those stages are *also*
  folded into the tiled pass (data-dependent, much larger rework).
- **The blur halo is large.** The incremental blur is a 5-deep chain, so a valid
  output stripe needs a cumulative ~25-row halo (Σ of per-level radii 3+4+5+6+7);
  at full octave-0 width the cache-fitting stripe is short, pushing halo recompute
  to 20–40%. The trade (redundant blur compute for saved bandwidth) is far less
  clearly positive than Tier 1's ~1-row halo.

So Tier 2 is deferred until Tier 1 is measured and (if pursued) tackled together
with folding orientation/description into the tiled pass.

### Diagnostics (environment variables)

These environment variables gate optional diagnostics (zero-cost when unset; all
output goes to stderr):

- `SFMTOOL_SIFT_NO_AVX2` — force the SSE2/scalar fallbacks everywhere (skip the AVX2+FMA
  paths), for A-B timing and reproducibility.
- `SFMTOOL_SIFT_TIMING` — emit per-stage wall-clock timing (`SIFT_TIMING build …` and
  `SIFT_TIMING detect …` lines).
- `SFMTOOL_SIFT_OPS` — emit one `SIFT_OP …` line per scale-space operator (`upsample`,
  `blur`, `decimate`) for offline aggregation. Independent of `SFMTOOL_SIFT_TIMING`.
  (No `dog` op: detection fuses the DoG per stripe, so it is never a standalone
  whole-image operator — see *tiled DoG/detect fusion*.)

One env var is a tuning knob rather than a diagnostic (it changes work
partitioning, not just logging), cross-listed here for discoverability:

- `SFMTOOL_SIFT_STRIPE_ROWS` — override the detect stripe height (default 32, clamped
  to `[1, 2^16]`); see *tiled DoG/detect fusion*. Output is unaffected.

### Extraction-orchestration pipelining

The Python extraction backend (`extract_sift_with_sfmtool`) processes images in
three stages: load+decode (`cv2.imread`), extract (the Rust core, GIL released,
internally rayon-parallel), then save (`write_sift`, zstd+ZIP). Measured
per-image stage split (ms): extract dominates at **85–91%** of the work;
load+thumbnail+save together are only **~7% on large images (2040×1536), ~15% on
small (270×480)**, and the source-file re-read used for content hashing is
effectively free (OS page cache serves it after the decode).

A load ∥ extract ∥ save pipeline is therefore a **single-digit-percent** win on
local/SSD storage, and is limited less by the GIL than by **CPU saturation**:
the rayon extract already uses every core, so overlapping the save — or decoding
the next image — *contends* for cores rather than hiding idle time. The one
stage with genuine idle time to hide is the **disk read** (`cv2.imread` already
releases the GIL) — *and*, on small images, the cores the per-image rayon extract
cannot itself saturate (measured: 1→4 threads scales only **3.1×** on a 270×480
image, ~23% idle), plus each image's serial floor (octave-0 build, setup). So
rather than a full three-stage pipeline, the backend implements several targeted,
low-risk overlaps:

- **Decode and extract several images concurrently**, yielding results in input
  order. A `ThreadPoolExecutor(max_workers=_extract_workers())` runs up to *K*
  decode+extract pipelines at once (FIFO over the in-flight futures preserves
  order and re-raises decode/extract errors in order); the bounded look-ahead
  caps memory to ~*K* decoded frames + pyramids. This hides disk-read latency
  **and** overlaps one image's serial floor with another's parallel work, so the
  cores the per-image rayon leaves idle on small images get filled. Because both
  `cv2.imread` and the Rust extract release the GIL and rayon's *single global
  pool* caps total CPU threads at the core count, more in-flight images never
  oversubscribe — they just keep that pool fed. *K* defaults to
  `min(os.cpu_count(), 4)` (override with `SFMTOOL_SIFT_EXTRACT_WORKERS`; `1`
  restores one-at-a-time). Measured win (point-in-time, 4-core box, release
  build, seoul_bull 270×480): **~20–25% batch throughput** (17 imgs 38→30 ms/img,
  100 imgs 35→29 ms/img), no single-image regression; the win grows on
  higher-core hosts where small-image per-image rayon is least efficient. (These
  are illustrative snapshots, not invariants — rerun `bench-sift` to refresh.)
  See `_stream_sift_with_sfmtool` and `_extract_workers` in
  `sift/extract_sfmtool.py`.
- **Stream `.sift` writes per image** instead of buffering a whole chunk
  (`chunk_size = 500` images) in memory — `extract_sift_with_sfmtool` is a
  generator that yields one result at a time, and `image_files_to_sift_files`
  writes each as it arrives. A peak-memory win (hundreds of MB for dense,
  high-resolution inputs). (Up front, before any extraction, images whose `.sift`
  is already newer than the source are skipped via an mtime check, so the
  pipeline only runs on stale/missing outputs.)
- **The `write_sift` binding releases the GIL** (`py.detach`) around its
  zstd/ZIP compression and file write, so a save can run concurrently with other
  Rust work.
- **Overlap the save with the next extract — on the rayon pool, not a thread.**
  `image_files_to_sift_files` drains results into a `SiftWriteQueue` (the
  `_sfmtool` PyO3 class): `submit` copies the data (GIL held) and `rayon::spawn`s
  the compression+write onto the **same global pool the extract uses**, then
  returns; `join_oldest` (bounded look-ahead for backpressure) and `join` await
  saves and surface their errors in order. The save of image *i* thus overlaps
  the extract of image *i+1*. This is **unconditional** — no spare-core gate.
  Backpressure bounds the queue at a **two-save look-ahead** (`write_lookahead`):
  before each `submit`, if two saves are already in flight the oldest is joined
  first, so at most ~2 compressed images are buffered. (Distinct from the
  *extract* concurrency's *K*-image look-ahead above.)
  The drain is guaranteed on every exit: the write loop's `finally` calls
  `drain` (a non-raising await of in-flight saves, so it can't mask an
  exception already unwinding), and `SiftWriteQueue::Drop` awaits any stragglers
  as a structural backstop — a spawned save never outlives the queue, so an
  error mid-stream can't leave a half-written `.sift` racing the next step.

Why both the decode and the save are worth hiding: they are **fixed per-image
costs that do not shrink with core count**, while the rayon extract does.
Measured on dino (2040×1536): decode ≈ 13 ms, save (zstd/ZIP) ≈ 26 ms — both
constant — versus extract ≈ 1244 / 730 / 546 / 407 ms at 1 / 2 / 3 / 4 threads
(Amdahl fit `≈ 165 + 1088/p` ms). So the decode+save "gap between images" is a
small slice when extract dominates (few cores, large images) but a growing
fraction as cores scale and extract collapses toward its ~165 ms serial floor.

**Why the save goes on the rayon pool, not a worker thread (the contention
trap).** Decode is I/O-bound (`cv2.imread` waits on disk and releases the GIL),
so prefetching it on a Python thread is free — it never burns a core. The save is
*CPU-bound* (single-stream zstd). An earlier attempt offloaded it to a dedicated
`ThreadPoolExecutor` writer thread; that **regressed ~25–30%** (wall *and*
CPU-seconds) on a fully-subscribed box. Root cause: a separate OS thread pushes
the runnable-thread count to *N+1* on *N* cores, so the kernel deschedules a
rayon worker mid-chunk; the descheduled worker stalls at one of the extract's
many sync barriers (sequential octaves, separable-blur passes) and the others
**busy-spin**, burning CPU for no work (hence the CPU-seconds inflation ≫ the
~2 s of actual save work). Verified: the same concurrent save caused **0%
slowdown with spare cores**, and external tenant load reproduced the identical
inflation — i.e. the cause is core oversubscription, not the write path.

Submitting the save as a **rayon task** instead avoids this: the pool stays at
*N* threads, one worker runs the ~26 ms save while the extract's `par_iter`
proceeds on the other *N−1* (rayon routes chunks only to available workers, so no
barrier waits on the saving worker → no spin). The cost is only the genuine
`save/N` of lost worker-time. Measured back-to-back on a 4-core box, the in-pool
overlap is **CPU-neutral vs inline** (the +25 % inflation of the thread approach
is gone), so it is safe to enable unconditionally; the wall-time win itself
materialises on many-core boxes, where the extract's serial floor leaves workers
genuinely idle for the save to fill. Ordering is preserved (the generator yields
in input order; `join`/`join_oldest` re-raise decode/write errors), and writes
target distinct files so a single in-flight save keeps up (save ≪ extract). The
COLMAP/OpenCV backends share the same queue unchanged (they return eager lists).

## Interface: split detection from description

**Yes — split keypoint finding from descriptor creation.** Recommended public API in
`crates/sfmtool-core/src/sift/`:

```rust
// Stage 1: detect + localize + assign orientation(s). Cheap and small per
// keypoint; returns a (potentially large) pool of oriented keypoints plus the
// scale space needed to describe any of them later.
pub fn detect_keypoints(image: &GrayImage, params: &SiftParams) -> Detection;

pub struct Detection {
    pub keypoints: Vec<SiftKeypoint>, // already oriented; sortable by size / response
    pub scale_space: ScaleSpace,      // retained Gaussian pyramid (DoG can be freed)
}

// Stage 2: describe an arbitrary *subset*, on demand. A descriptor is a pure
// function of (scale_space, keypoint), so this can be called for as few or as
// many keypoints as you need, in any order, any number of times.
pub fn compute_descriptors(scale_space: &ScaleSpace, keypoints: &[SiftKeypoint]) -> Descriptors;
pub fn compute_descriptor(scale_space: &ScaleSpace, keypoint: &SiftKeypoint) -> [u8; 128];

// Convenience one-shot that builds the pyramid once and runs both, describing
// every keypoint (the common case / describe-all path).
pub fn extract_sift(image: &GrayImage, params: &SiftParams) -> SiftFeatures;

// Same, but describe only the top-`max_described` keypoints (the size-sorted
// prefix); every keypoint is still returned, only the descriptor count is
// capped. `None` describes all. Lets you detect a large pool cheaply and pay
// descriptor cost for just a small working set in a single call.
pub fn extract_sift_partial(image, params, max_described: Option<usize>) -> SiftFeatures;
```

Raising the *detection* cap is nearly free: detection is dominated by
extrema-finding and sub-pixel localization, which scan the whole image
regardless of `max_num_features`; only orientation scales with keypoint count.
Describing fewer keypoints scales descriptor cost down ~linearly (~0.01 ms/kp).
So `extract_sift_partial` with a high `max_num_features` and a small
`max_described` (e.g. detect 16384, describe 1024–2048) runs ~10% faster than
describing all 8192 on dense images and registers the same images end-to-end,
while leaving a large keypoint reservoir to describe on demand later — at the
cost of a proportionally sparser point cloud (≈29% of the points at 2048, ≈14%
at 1024 on `dino_dog_toy`, with full registration in all cases).

```rust
pub struct SiftKeypoint {
    pub x: f32,        // refined, full-resolution coords; COLMAP pixel-center convention (0.5, 0.5)
    pub y: f32,
    pub affine_shape: [[f32; 2]; 2],  // [[a11, a12], [a21, a22]] like COLMAP; a scaled rotation for SIFT
    pub octave: i32,   // for descriptor sampling at the right pyramid level
    pub layer: f32,    // sub-level within octave
    pub response: f32, // |D(x̂)| contrast, for ranking / nfeatures cap
}

pub struct SiftParams { /* table above; ::default() = Lowe 2004 */ }
pub struct SiftFeatures { pub keypoints: Vec<SiftKeypoint>, pub descriptors: Descriptors }
```

Why split:

- **Detect broad, describe narrow.** Detection is cheap and a keypoint is tiny
  (a handful of floats), so we can afford to detect a *large* pool — lower the contrast
  threshold, keep every octave — while a descriptor is 128 bytes and the heaviest
  per-keypoint compute (rotated 16×16 sampling + trilinear scatter + normalization).
  Splitting lets the descriptor cost (and storage) scale with what's actually used,
  not with how many keypoints were found. See *Lazy descriptors* below.
- **The pyramid is the shared, expensive artifact.** Both stages read the same
  Gaussian `ScaleSpace`; building it once and passing a borrow avoids recomputation —
  exactly how DIS builds pyramids once and reuses them.
- **Mirrors OpenCV's `detect` / `compute` / `detectAndCompute`** split, so the Python
  backend swap is mechanical.
- **Independent parallelism.** Detection parallelizes over image tiles/levels;
  description parallelizes over keypoints. Splitting lets each use its natural grain.

A `ScaleSpace` (Gaussian + DoG pyramids) type is the natural shared handle, analogous
to `ImagePyramid` in optical flow. `extract_sift` constructs it, calls both stages, and
is what the PyO3 binding wraps.

### Lazy descriptors and coarse-to-fine

The split is not just a code-organization nicety — it enables a **detect-many,
describe-few** workflow that the dense `extract → 128 bytes per keypoint → match`
pipeline can't express.

**Keypoints are self-describing.** A `SiftKeypoint` carries everything the descriptor
needs — `(x, y)`, the `affine_shape` (encoding scale and orientation), and the
`octave`/`layer` that pick the Gaussian level — so `compute_descriptor(scale_space, kp)` is a pure, idempotent
function with no hidden detection state. Detect once, then realize descriptors for any
subset, in any order, whenever you need them.

**Coarse-to-fine by keypoint size.** SIFT's octave structure *is* a size hierarchy:
coarse octaves yield a few large-`σ` keypoints over broad structure; fine octaves yield
many small-`σ` keypoints. Because each keypoint knows its `scale`, the pool sorts by
size for free. A matcher can then:

1. Describe and match only the **largest / highest-response** keypoints to establish a
   rough correspondence or pose. Few keypoints → few descriptors.
2. Descend to finer scales, computing descriptors **only where they help** — inside the
   image regions or epipolar bands the coarse stage already made plausible — instead of
   describing the whole fine-scale population up front.

This trades descriptor compute for a cheap pre-pass over keypoint *metadata*, and the
work scales with the precision you actually need rather than the total keypoint count.

**Lazy fill pairs with the existing flow matcher.** `feature_match/_flow_matching.py`
already advects keypoints through optical flow and then descriptor-matches each against
only its spatial KdTree candidates. With lazy descriptors, the advection and spatial
culling run on bare keypoints, and descriptors are computed **only for the surviving
candidate set** — so the cost follows the candidate count, not the keypoint count.
The same holds for epipolar-guided or covisibility-pruned matching.

**What this costs.** Lazy fill means retaining the `ScaleSpace` (Gaussian pyramid)
between detection and description — memory, not recompute. The DoG pyramid is only
needed for detection and can be dropped immediately; descriptors read the Gaussian
levels (or their precomputed gradient magnitude/orientation). Retention is therefore a
deliberate trade-off the caller opts into: `extract_sift` builds, uses, and frees the
pyramid in one shot, while `detect_keypoints` hands the `ScaleSpace` back so the caller
controls its lifetime (drop it, or keep it — e.g. behind an `Arc` — to lazily describe
later). For very large images the pyramid can be rebuilt per-octave on demand, or the
cached representation narrowed to per-level gradients, if the memory cost outweighs the
recompute savings — an implementation detail to settle with benchmarks.

### On-disk incremental extraction (`.sift` format extension)

Lazy descriptor fill must survive **across CLI commands** — one command detects the
keypoint pool, later commands describe more of it on demand — so the working state lives
on disk, not just in memory. We extend the `.sift` archive itself into a **growable**
container rather than adding a sidecar. The design rests on one structural choice:

**The `.sift` format already stores features sorted by descending size** (the existing
backends do this), and the incremental design depends on that ordering. Coarse-to-fine
always wants the largest keypoints first, so the set of *described* keypoints is always a
dense **prefix** `[0, M)` of the keypoint list. That
means we never need a sparse coverage mask — coverage is a single integer `M`
(`described_count`), and descriptors are stored as contiguous **range chunks** that tile
that prefix. (Sort ties broken deterministically — e.g. by response, then octave, then
`(y, x)` — so the order is reproducible.)

**Descriptor chunks as range-named tensor entries.** Instead of one `descriptors`
array, the archive holds a sequence of append-only chunk entries named by their
inclusive keypoint-index range:

```
descriptors.0-100.128.uint8       # first describe-batch: keypoints 0..=100  (101 rows)
descriptors.101-1000.128.uint8    # later batch appended:  keypoints 101..=1000 (900 rows)
descriptors.1001-4095.128.uint8   # ...
```

Chunks are contiguous and gap-free (`next.start == prev.end + 1`) because coverage only
ever grows as a prefix. Reading the full descriptor block = concatenate chunks in order;
reading the top-K (coarse-to-fine) = read only the chunks covering `[0, K)`, the natural
extension of `read_sift_partial`.

**Append data, rewrite two small files.** Each describe-batch:

1. **Appends** one new immutable `descriptors.<a>-<b>.128.uint8` entry to the ZIP. Bulk
   data is strictly append-only and never rewritten.
2. **Rewrites the two small mutable JSON entries** — `features/descriptors_metadata.json`
   (the coverage count `described_count`) and `content_hash.json` (hashes only).
   `metadata.json` and the keypoint/thumbnail arrays stay immutable, so the stable
   `feature_set_xxh128` and `metadata_xxh128` never change. Both rewritten files are tiny.

The integrity model evolves but stays verifiable. Today's `content_xxh128` is already a
*digest of digests* (it hashes the concatenation of each array's xxh128). We keep that
structure: `content_hash.json.zst` lists the per-entry digests, including one per
descriptor chunk, and `content_xxh128` is the hash of the concatenated digest list.
Appending a chunk therefore only appends one digest and rehashes the small digest list —
**no rehashing of existing data**. At every point the archive is fully verifiable; the
contract changes from "exactly `N` keypoints each with a descriptor" to "`N` keypoints
with descriptors for a verified prefix of length `described_count ∈ [0, N]`."

**Lifecycle / CLI flow** (proposed — `--detect` / `--describe` / `--top-k` are
not yet implemented; today `sfm sift --extract` writes a fully-described file):

```
sfm sift --detect images        # writes keypoints sorted by size; described_count = 0
sfm sift --describe -i images --top-k 1000   # appends descriptors.0-999; described_count = 1000
sfm match ...                   # triggers describe-on-demand for the keypoints it needs,
                                # appending further chunks; reuses any already on disk
```

A later command reads `described_count`, computes only the still-missing descriptors
(rebuilding the `ScaleSpace` from the source image — deterministic given params and the
`image_file_xxh128` already recorded), and appends them. The pyramid rebuild is the
price of cross-process laziness; it is paid only when new descriptors are actually
needed, and amortizes when a batch describes many keypoints at once.

**External consumers.** Tools that need a complete dense block (COLMAP export, bulk
matching) require `described_count == feature_count`, i.e. describe-all. Once fully
described, a v2 file can simply be written in the v1 layout (a single descriptors array) for
those consumers — that is an ordinary format conversion, not a special operation.

**Referential stability.** Appending descriptors changes the file's whole-file
`content_xxh128`, which would break any `.sfmr`/`.matches`/workspace reference that pinned
the `.sift` by that hash — even though the expanded file is a strict superset. The format
solves this with a **stable** `feature_set_xxh128` (over the immutable image + keypoints +
tool config, excluding descriptors) that references should use instead. Descriptor-dependent
consumers (matches) additionally verify the immutable `[0, M)` descriptor prefix they relied
on. Full definitions in [`../formats/sift-file-format.md`](../formats/sift-file-format.md).
This implies updating the `.sfmr`, `.matches`, and workspace specs to reference
`feature_set_xxh128`.

**Concurrency.** Appending to one ZIP plus rewriting its metadata is a single-writer
critical section. Wrap describe-and-append in an advisory file lock on the `.sift`
(monotonic prefix growth makes the lock window short and conflicts rare). Concurrent
*readers* are unaffected: they read `described_count` and the chunks present.

**Open implementation details to validate.**

- ZIP append mechanics with the `zip` crate: appending data entries is cheap (rewrites
  only the central directory), but *replacing* the two mutable JSON entries
  (`features/descriptors_metadata.json.zst`, `content_hash.json.zst`) needs either
  duplicate-name-last-wins or a central-directory rewrite — pick and pin the reader's
  resolution rule.
- This is a `.sift` **format version bump**; `read_sift` must handle both the legacy
  single-`descriptors` layout and the new chunked layout (legacy = one implicit
  `0-(N-1)` chunk with `described_count = N`).
- The normative on-disk definition (chunk naming grammar, `described_count`, the
  `component_xxh128` digest cache and recompute rule, concurrency) lives in
  [`../formats/sift-file-format.md`](../formats/sift-file-format.md) under "Incremental
  descriptor extraction (version 2)"; this section is the design rationale.

### Module structure

```
sfmtool-core/src/sift/
├── mod.rs          # Public API: SiftParams, SiftKeypoint, extract_sift, detect_keypoints, compute_descriptors
├── scale_space.rs  # ScaleSpace: Gaussian pyramid + DoG (separable blur, octave downsample)
├── detect.rs       # 26-neighbor extrema + subpixel localization + contrast/edge rejection
├── orientation.rs  # gradient precompute + 36-bin histogram + multi-peak assignment
└── descriptor.rs   # 4x4x8 trilinear-interpolated descriptor + normalize/clamp/quantize
```

### Python bindings

`crates/sfmtool-py/src/py_sift.rs`, registered in `sfmtool-py/src/lib.rs`, following
`py_optical_flow.rs` conventions (`PyReadonlyArray2` in, `IntoPyArray` out,
`py.detach(...)` around the compute):

- `detect_sift_keypoints(image, params=None) -> (positions (N,2) f32, affine_shapes (N,2,2) f32, responses (N,) f32)`
- `extract_sift(image, params=None) -> (positions (N,2) f32, affine_shapes (N,2,2) f32, descriptors (N,128) u8)`

This output is exactly what `src/sfmtool/sift/` already consumes, so a new `extract_rust.py`
backend slots in alongside `extract_opencv.py` / `extract_colmap.py` and writes via the
existing `sift-format` path. The affine shapes pass straight through — no conversion needed,
unlike the OpenCV backend, which derives them from its `KeyPoint`s via
`opencv_keypoint_to_affine_shape`.

## Phasing

1. **Phase 1 (this spec): CPU + SIMD + multithread.** Scale space, DoG, detection,
   localization, orientation, descriptor; SSE2 + rayon. Cross-validate against OpenCV
   SIFT on the test datasets.
2. **Phase 2 (future):** GPU compute shaders (separable blur, DoG, extrema, descriptor),
   reusing the wgpu infrastructure — documented in a future `specs/core/gpu-sift.md`.

## Testing & validation

- **Cross-validation against OpenCV** (`cv2.SIFT_create`) on the checked-in datasets
  (`seoul_bull_sculpture`, `seattle_backyard`, `dino_dog_toy`): keypoint count in a
  sane band, location repeatability (≥X% of OpenCV keypoints matched within 1–2 px and
  √2 scale), and descriptor match rate / mean cosine similarity over mutual nearest
  neighbors above a threshold. Thresholds calibrated like the optical-flow table.
- **Synthetic invariance tests:** rotate/scale a test image, confirm keypoints
  re-detect at the transformed locations/scales and descriptors match (the paper's own
  evaluation methodology).
- **Rust unit tests:** known-extremum toy DoG, edge-response rejection on a synthetic
  ridge, descriptor unit-norm + 0.2-clamp invariants, orientation on a synthetic
  gradient.
- **PyO3 surface test** (`tests/rust_bindings/test_sift_extract_rust_bindings.py`) exercising the bindings and
  round-tripping through `sift-format`.
- **Criterion benchmarks** (`crates/sfmtool-core/benches/sift.rs`): pyramid build,
  detection, descriptor, end-to-end — same structure as `benches/optical_flow.rs`.

## Dependencies

No new crate dependencies anticipated: `rayon` for parallelism, the existing
`sift-format` crate for I/O, `criterion` (dev) for benchmarks. SIMD via `std::arch`.
