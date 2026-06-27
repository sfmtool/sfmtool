# Per-View Cache + SIMD Search for Keypoint Localization

_Status: **proposed** (design). Accelerates the **integer** cross-view search in
[patch-keypoint-localization.md](patch-keypoint-localization.md) (the congealing
algorithm) — the step that dominates `sfm embed-patches`. Scope here is integer
refinement only: the discrete search, view selection, and consensus. Its job is
to land each keypoint **in the basin** (the right integer cell) — getting it to
true sub-pixel is a **separate, independent algorithm**,
[keypoint-subpixel-refinement.md](keypoint-subpixel-refinement.md), that consumes
this one's output as a seed. The plan is the **regular full-resolution integer
grid**, made cheap by SIMD — not a reduced-resolution search; a resolution
multiplier exists as a knob but isn't the expected path (see "Search resolution
multiplier"). Models its structure on the
[fronto-parallel patch cache](fronto-parallel-patch-cache.md) ("render once,
score many"). The kernel lives in
`sfmtool-core/src/patch/keypoint_localize.rs`; opt-in phase timing is in
`keypoint_localize/prof.rs`._

> _Already landed (branch `optimize-search-shift`): the per-candidate
> extract→z-normalize→dot search was reformulated as **correlation
> accumulation** — three maps per channel (`Σ kern·I`, `Σ w·I`, `Σ w·I²`) whose
> inner loop is a contiguous SAXPY — proven equivalent to the per-candidate path
> by `search_shift_ref` and the `search_shift_matches_reference*` tests. That
> alone cut `search_shift` ~2.5× (602→241 µs/call) and `localize` wall ~2.1×.
> This spec is the next step: a render-once per-view cache and a hand-rolled AVX2
> kernel._
>
> _**Phase 1 landed** (branch `keypoint-localization-cache`): the **per-view
> render-once cache** (sized `R + 4·search`) plus **integer-tracked reads**. The
> congealer now renders each view's frame-oriented cache once and every round
> reads its core and scores its shift grid from that cache at the view's current
> **integer** offset `iacc`; the parabolic sub-pixel residual is tracked alongside
> (convergence + final keypoint) but never folded back into the read position, so
> every cache read is exact. `render_context` collapses from per-(view, round) to
> per-view. The `search_resolution_multiplier` knob (default `1.0`, a no-op) is in
> place. Still scalar — the centered-`f32`/planar AVX2 kernel and the `i16` path
> are later phases. The `search_shift` accumulation math is unchanged (the
> `search_shift_matches_reference*` equivalence tests still pass, now also
> asserting integer-argmax agreement). Integer-tracked reads build the consensus
> from integer-aligned cores, a deliberate sub-px change from the
> fractionally-rendered reference; the existing congealing tests pass unchanged._

## Problem

During keypoint localization the patch frame and normal are **fixed**
(refinement already ran); only each view's in-plane offset moves across rounds.
Yet the current loop, per round, **re-renders** each view's `cr×cr` context tile
from the source (`render_context`) and then slides an integer windowed-ZNCC
search over it (`search_shift`). On the dino reconstruction (85 imgs @
2040×1536, 19 024 pts):

| Phase | Share of `localize` | Notes |
|---|---|---|
| `search_shift` | 55.9% | 1.44 M calls, 241 µs each (post-reformulation) |
| `render_context` | 32.9% | one `cr×cr` warp+remap per view **per round** |
| `loo_template` | 10.4% | leave-one-out consensus build |

Measured sizes (the defaults `KeypointLocalizeParams` carries): `R = 24`
(resolution), `search = 6`, `margin = ⌈search⌉ = 6`, `span = 2·margin+1 = 13`
(the `13×13 = 169` shift grid), `cr = R + 2·margin = 36`, support `n ≈ 452` (the
`GaussianDisk` unit disk over `R²`), `channels ≈ 3` (RGB). The reformulated
search does `channels·n·span²·3 ≈ 0.69 M` FMA per call at 241 µs ≈ **~5.7
GFLOP/s — ~25% of one core's f64 AVX2 peak**, i.e. the inner loop is *not* well
vectorized. There is ~4× headroom in f64, ~8× in f32, plus the renders are a
second large slice that is mostly redundant work.

## Idea

Two compounding changes:

1. **Render one expanded cache per view, once.** The base the whole round loop
   reads from — both the leave-one-out consensus build *and* every round's
   search — instead of re-warping per round.
2. **Hand-roll the search as a register-blocked AVX2 kernel** over that cache,
   in centered `f32` (stage 1), with an integer `i16` variant evaluated as a
   follow-up (stage 2).

### Why render-once is exact, not an approximation

During localization the patch `(center, u, v, normal)` is fixed; the only
per-round change is the in-plane offset (patch-grid px). A grid pixel `(g)`
rendered at offset `δ` samples the source at `project(center + (g+δ)·axes)` —
which is exactly the cache sample at grid position `g+δ`. So **reading the cache
at an integer-shifted position is bit-identical to re-warping the patch at that
offset.** The current per-round re-render recomputes those same samples.

This exactness holds only for **integer** shifts, so the congealer tracks an
**integer** read accumulator `iacc` (the cache index it reads consensus cores
and search candidates from) separately from the **sub-pixel residual** the
parabolic estimate accumulates. The residual feeds convergence detection and the
hand-off seed (below) but is **never folded back into the read position** — if it
were, the reads would become fractional and we would forfeit the exactness (back
to a bilinear resample per candidate). With integer-tracked reads, *every* cache
read in the loop is exact; the cache is an exact restructuring, not a
fronto-cache-style affine approximation. (The accurate sub-pixel offset is then a
separate algorithm — see "Sub-pixel hand-off" below.)

## The cache

Per view, render once at the seed (`acc = 0`, i.e. centered on `project_i(X_p)`):

- **Geometry / size.** The window centre can reach `acc + d` where `|acc| ≤
  search` (the clipped accumulated drift) and `|d| ≤ margin` (the in-round
  search), so it spans `±2·search`; the window itself extends `±R/2`. The cache
  must cover `R + 4·search` per axis (`= 48` at the defaults). The frame
  orientation is the patch's fixed `(u, v)`; pixels map to the patch grid 1:1
  (no supersampling — see below).
  - *Alternative (smaller cache):* clamp the **search window** to a global
    `±search` (not just the accumulated `acc`). Then the centre stays in
    `±search`, the cache is `R + 2·search = 36` (today's tile size), and one
    render still covers every round. This is a minor change to the search
    semantics (the probe can no longer transiently exceed `±search` mid-round)
    and must be validated against the reference; if equivalent, prefer it.
- **Rendering.** The same `WarpMap::from_patch` + `remap_*` path
  `render_context` uses, just over the expanded grid and once per view. The
  source pyramid is touched **once per view** (today: once per view per round).
- **Format / layout (stage 1).** Planar **per channel**, **centered** `f32`:
  `plane[c][row·istride + col]`, value `I − c̄_c` where `c̄_c` is that channel's
  mean over the cache. Centering is load-bearing (next section). The row stride
  is padded so a 16-wide aligned load from any support column is in bounds:
  `istride = align_up(cacheW − 1 + 16, 8)`; pad columns hold `0` (= the mean,
  harmless — they only feed discarded grid cells).
- **Validity.** One per-pixel invalidity plane (`1.0` out of frame else `0.0`),
  built with the cache. A search window with any invalid support pixel is
  unscorable (`−∞`), exactly as `extract_core` returning `false` today.

## Round loop with the cache

```
once per view:   render the view's expanded cache (planar centered f32 + validity)
each round:
   per view:   read the current R×R core from its cache at integer iacc   (no render, exact)
               z-normalize → contribute to the leave-one-out consensus
   per view:   integer search within its cache vs the LOO consensus → grid
               argmax → integer δ_int; parabolic(grid) → δ_sub
               iacc += δ_int   (clip);   residual_v = δ_sub
   drop failing views (max_shift_px / LOO-ZNCC / out-of-frame)
   converge when mean |δ_int + δ_sub| < convergence_px
```

Both the consensus-core read and the search candidates are at **integer** cache
positions (`iacc`, `iacc + d`), so they are mutually aligned and exact. `δ_sub`
is the parabolic estimate off the discrete grid; it gates convergence and seeds
the sub-pixel hand-off, but does not move `iacc`. Reads are L1 hits (a view's
cache is `cacheW²·channels·4 B ≈ 27 KB` at `cacheW = 48`, ~15 KB at 36). The
`render_context` phase collapses from per-(view, round) to per-view.

Building the consensus from integer-aligned cores (vs the reference's
fractionally-rendered cores) is a sub-px behaviour change — see Validation.

## Search kernel (stage 1: centered f32, register-blocked AVX2)

Windowed ZNCC over the shift grid, per kept channel `c`, as three correlation
maps (already implemented scalar; this is the AVX2 form):

```
Ncross_c(s) = Σ_k kern_c[k]·I_c[s+k]     kern_c[k] = √w[k]·tmpl_c[k]
S1_c(s)     = Σ_k w[k]·I_c[s+k]
S2_c(s)     = Σ_k w[k]·I_c[s+k]²
zncc_c(s)   = (Ncross_c − mean_c·Σ√w·tmpl_c) / √(S2_c − S1_c²/W)   (0 if var < FLAT_EPS)
ZNCC(s)     = (1/channels) Σ_c zncc_c(s)        mean_c = S1_c/W
```

(`s` runs over the `span×span` grid; `W = Σ w`. The mean term is carried
explicitly so the result is algebraically identical to z-normalize-then-dot for
any template.)

**Register-blocked loop**, order **channel → grid-row `gy` → support `k`**,
holding the row's accumulators in registers across the `k`-loop so the image
streams through once and the grids never round-trip to memory in the hot loop:

```
for c in channels:                         // plane_c (centered), kern_c, w, Σkern_c
  for gy in 0..span:
    n_lo=n_hi=s1_lo=s1_hi=s2_lo=s2_hi = 0  // 6 YMM accumulators (2× __m256 per map)
    for k in 0..n:
      off = (gy + r_k)·istride + c_k
      src_lo = loadu(plane_c[off..]); src_hi = loadu(plane_c[off+8..])   // 16 cols
      kb = bcast(kern_c[k]); wb = bcast(w[k])
      n_lo  = fma(kb, src_lo, n_lo);  n_hi  = fma(kb, src_hi, n_hi)
      s1_lo = fma(wb, src_lo, s1_lo); s1_hi = fma(wb, src_hi, s1_hi)
      sq_lo = mul(src_lo,src_lo); sq_hi = mul(src_hi,src_hi)
      s2_lo = fma(wb, sq_lo, s2_lo); s2_hi = fma(wb, sq_hi, s2_hi)
    combine this row's `span` cells → zncc_c, add into the combined grid
```

Inner loop: **2 loads, 2 mul, 6 FMA, 2 broadcasts** for 16 lanes; ~12 YMM live
(≤16). The three maps share each `src` load (high arithmetic intensity); the
`span→16` padding wastes ~19% lanes (acceptable). The whole plane is in L1, so
every load hits L1.

**Combine** (per row, after the `k`-loop): `zncc_c` for the row's `span` cells
(scalar, or AVX2 `rsqrt` + one Newton step over `span` lanes), accumulated into
the combined grid. `~span²·channels` cells — small.

**Validity**: a separate single-accumulator pass over the invalidity plane (same
structure, channel-independent) masks `−∞`. **Argmax + separable parabolic
sub-pixel** are unchanged from `search_shift_ref`.

### Why centering enables f32

The denominator `S2 − S1²/W` is a catastrophic-cancellation trap in `f32`
(`S2 ~ 10⁷`). Centering the cache by the per-channel mean makes `S1 ≈ 0` and
`S2 ≈ variance·W` (small), so the cancellation vanishes and `f32` is accurate —
which buys the 8-lane width over 4-lane `f64`. The numerator is recovered
exactly: `Ncross = Ncross' + c̄·Σkern`. Centering is the layout decision that
makes the `f32` kernel both fast and correct.

## Sub-pixel hand-off

This spec covers **integer refinement only**: the discrete cross-view search,
view selection, and consensus, accelerated by the cache. The parabolic estimate
it computes stays *inside* the integer solve — it detects convergence and seeds
the next stage — but the cache reads remain integer (above), so this spec never
produces an accurate fractional keypoint.

The accurate sub-pixel offset is a **separate algorithm**: a continuous
photometric (ECC / Lucas–Kanade) solve that optimizes the image objective with
gradients, run once after this converges, seeded by `iacc + residual`. It is
specified in
[keypoint-subpixel-refinement.md](keypoint-subpixel-refinement.md). Keeping it
separate is what lets the integer search stay integer-exact (and lets stage 2's
`i16` path work — it never has to resample fractionally).

## Search resolution multiplier (`f32`) — an available knob, not the plan

The expectation is to run the **full-resolution** integer grid (`R`) and let SIMD
carry the cost; we don't expect to reduce resolution. This knob is documented as a
fallback in case SIMD isn't enough, not as the intended path.

A configurable multiplier `m` (`f32`, **default `1.0`**) sets the **search
resolution** `R_s = round(m·R)`: the cache, support, window, and shift grid are
built at `R_s`. Its cost scales with the support count `n ∝ R_s²` (and the grid
`span_s`), so `m < 1` is a smooth speed fallback — `m = 0.5` quarters `n`
(~452 → ~113). It's safe here only because of the two-stage split: the grid search
just has to land the right **integer cell**, and the separate fine tune recovers
true sub-pixel accuracy at full resolution; a lower `R_s` smooths the ZNCC surface
but keeps the peak within `1/m` patch-px of the true offset.

The multiplier is **orthogonal to the search strategy** — it changes the
*correlation resolution*, not which candidates are visited (not coarse-to-fine
pruning). Units: an integer step in the `R_s` grid is `1/m` patch-px, so the found
shift is scaled by `1/m` back to patch-px for the accumulator and the fine-tune
seed. The cache's integer-shift exactness holds unchanged at `R_s`.

Expose it as `KeypointLocalizeParams { search_resolution_multiplier: f32 }`,
default `1.0`; only reach for `m < 1` if profiling shows the full-resolution SIMD
grid is still too slow.

**`m > 1` is a faster, *discrete* sub-pixel approximation — a cheaper option, not a
replacement for the high-accuracy refiner.** At `m > 1` an integer step in the
`R_s` grid is `1/m < 1` patch-px, so the supersampled grid resolves sub-pixel
offsets **directly** in one all-SIMD kernel (no gradients / consensus-refresh /
fractional sampling), at ~`m²`-ish more search work. But it stays *quantized* to
the grid; the continuous LK fine tune
([keypoint-subpixel-refinement.md](keypoint-subpixel-refinement.md)) reaches the
true optimum and remains the high-accuracy / ground-truth option. So they
**coexist**: the supersampled grid when its accuracy is good enough and speed
matters; the LK fine tune when quality matters most. Where that line falls is a
measurement question.

## Open topic: search strategy (dense vs. non-exhaustive)

Whether the grid search visits **every** cell is an open design axis, not a
settled decision. The baseline here is a **dense** grid because the
correlation-accumulation kernel computes the whole grid efficiently — the support
reads amortize across all candidates — which is especially attractive when the
grid is small (low `m`). But the search need not be exhaustive.

The choice is **coupled to the kernel**:

- **Dense grid → accumulation kernel.** Cost is ~independent of how peaked the
  surface is; amortized, SIMD-friendly, robust (can't miss a cell). Best when the
  grid is small.
- **Non-exhaustive (hill-climb / local / early-out) → per-candidate scoring.**
  Visits few candidates but forfeits the amortization, so each candidate pays the
  full support gather. Wins only if it visits *few enough* candidates to beat the
  dense kernel's amortized total — and it risks landing in a wrong local peak,
  which the fine tune (a *local* solve) cannot rescue.

The earlier rejection of coarse-to-fine was specifically that it was the wrong
*SIMD lever* (it prunes candidates but leaves each scalar and risks the peak);
with the cache + accumulation, dense became cheap. A non-exhaustive strategy
reopens as a *separate* axis now — and it could even be **round-dependent** (a
wider search in round 1, a tight local search once the offset is near-converged).

Defer the decision: start dense (simplest, matches the kernel), measure at the
chosen `m`, and revisit a non-exhaustive strategy only if the grid search is
still hot — guarding robustness, since the fine tune does not backstop a missed
cell.

## Stage 2 (follow-up): integer `i16` correlation — investigated, dropped

The source is `u8`, so in principle the cache can be `u8`/`i16` and the hot
accumulation can use integer SIMD (`_mm256_madd_epi16` fuses mul+add at 16 `i16`
lanes; `_mm256_sad_epu8` sums `u8` nearly free) — potentially 2× the f32 lanes.
`Ncross` (`Σ kern_q·I`) and `S1` (`Σ I`) integerize cleanly.

**The snag is `S2 = Σ w·I²`**: `I²` is 16-bit and the per-pixel weight does not
fuse into `madd`, and an `i32` accumulator can overflow over ~450 pixels. The
clean integer route normalizes with a **box window** (`Σ I`, `Σ I²` unweighted
via `sad`/`madd`) while keeping the Gaussian weights only in the numerator
kernel — an *approximation* of the ZNCC denominator. It is no longer
bit-equivalent to the reference, so stage 2 is gated on **argmax agreement**
(does it pick the same shifts?) and a real speedup over stage 1, not a tolerance.

> _**Investigated 2026-06-27 (branch `keypoint-localize-i16`, not merged).**_
> A prototype implementation (scalar reference + AVX2 with `vpmulld`/`vpaddd`
> over i32 lanes, plus a u8 cache plane and an `SFMTOOL_LOCALIZE_KERNEL`
> dispatcher in `f32` / `i16` / `compare` modes) was benchmarked head-to-head
> against stage 1 on dino (18 961 points). **Both gates failed:**
>
> - **No speedup.** `search_shift` i16: **98.7 µs/call** vs stage-1 f32 **93.6
>   µs/call** (~5% slower). Root cause: `vpmulld` is 5c/2c vs FMA at 4c/0.5c,
>   and the i32-lane design (8 cells/half, same lane count as f32) doesn't
>   recover throughput against the FMA-saturated f32 inner loop. The "2× lane
>   count" potential needs the trickier `_mm256_madd_epi16` horizontal-pair-sum
>   design, which the prototype didn't attempt because the surrounding gather
>   pattern (now ~88% of `search_shift` per the new `search_acc` sub-phase
>   timer) is already the bottleneck and the multiplies aren't the dominant
>   cost.
> - **14.67% argmax disagreement** in compare mode (1 410 766 agree /
>   242 472 disagree out of 1 653 238 calls). The kept point count is
>   nevertheless identical to f32 (18 961) — the multi-round congealing
>   loop absorbs per-call disagreements via parabolic residuals and view-
>   selection gates. Acceptable in isolation but no speedup to trade for it.
>
> The investigation's permanent residue in the codebase is just the
> `search_acc` / `search_combine` / `search_argmax` sub-phase timers in
> `keypoint_localize::prof`, which proved valuable in isolating where time
> goes inside `search_shift` and would inform any future revisit. The i16
> kernel itself, the u8 cache plane, the dispatcher, and the compare-mode
> agreement counters were not committed — they would have bloated the f32 hot
> path (the u8 plane is built every render at ~20 KB/view extra) for no
> measured benefit. Future revisit should start from the `madd_epi16` lane-
> packing design or take a different angle (e.g. attack the gather pattern).

## Numerical fidelity & validation

Two levels — the **search kernel** is provably equivalent; the **congealing loop**
is a deliberate sub-px behaviour change:

- **Search kernel (equivalent).** `score_grid` is algebraically identical to the
  per-candidate `extract → z-normalize → dot` for a given tile + template;
  validate with a relative tolerance (~1e-3) on the ZNCC grid and **exact argmax**
  on clear-peak fixtures (templates built from the tile's own core at a known
  shift, as in `search_shift_matches_reference`).
- **AVX2 vs scalar.** Keep the centered-`f32` scalar form as the reference and
  add an `avx2_matches_scalar` test within `f32` tolerance (mirrors the fronto
  cache's `resample_avx2_matches_scalar`). The scalar form is also the
  non-x86 / non-AVX2 fallback (runtime-dispatched, like the cache).
- **Congealing loop (argmax/selection agreement, not bit-equivalence).**
  Integer-tracked reads build the consensus from integer-aligned cores rather
  than the reference's fractionally-rendered ones, so the loop is no longer
  bit-equivalent. Validate that it picks the **same kept views and the same
  integer registrations** as `search_shift_ref`-based congealing on the datasets,
  and that the integer-plus-residual offsets agree within ~1 px.
- **End-to-end.** `sfm embed-patches` on dino keeps a sane point count and the
  registrations are as good or better; re-profile with `SFMTOOL_PROFILE=1` to
  confirm the `render_context` collapse and the `search_shift` speedup. (The
  final keypoint *accuracy* is owned by
  [keypoint-subpixel-refinement.md](keypoint-subpixel-refinement.md), validated
  there.)
- The existing 19 keypoint-localization kernel tests must continue to pass.

## Design choices (and why)

- **Render once per view, not per round.** The frame is fixed during
  localization; integer in-plane shift = integer cache index shift, so one
  expanded render reproduces every round's integer-shift samples exactly.
  Removes the redundant per-round warp (the 33% `render_context` slice).
- **Centered `f32`.** Removes the variance cancellation that would otherwise
  bar `f32`, unlocking 8-lane width with no accuracy loss.
- **Register-blocked on the grid row.** Keeps the search's hot loop pure
  register FMA with the image streamed once — the structural fix for the ~25%
  ALU utilization the profile shows.
- **Integer-only search; sub-pixel is a separate algorithm.** Tracking integer
  reads keeps every cache access exact and lets the `i16` path avoid fractional
  resampling entirely; the parabolic estimate stays inside the loop only to seed
  and detect convergence. Accuracy is owned by the continuous solve
  ([keypoint-subpixel-refinement.md](keypoint-subpixel-refinement.md)). No
  supersampling — it would inflate the bottleneck for sub-pixel produced better
  elsewhere.
- **Cache `R + 4·search` vs `R + 2·search`.** The larger cache is unconditionally
  correct for the current incremental-clip search; the smaller one needs a
  search-window clamp and validation. Prefer the smaller if it proves equivalent.

## Open questions

- Cache size: confirm the `±search`-clamped `R + 2·search` cache is equivalent
  to the reference, or keep `R + 4·search`.
- Centering constant: per-channel cache mean (best conditioning) vs a fixed
  `127.5` (cheaper). Measure whether the fixed constant is accurate enough.
- Combine step: scalar vs AVX2 `rsqrt` — only worth vectorizing if it shows up
  after the accumulation is sped up.
- Stage 2 box-window normalization: does the argmax track the Gaussian-window
  reference closely enough on the datasets to justify the integer speedup?
- Search resolution multiplier `m < 1`: only relevant if the full-resolution SIMD
  grid is too slow. If it comes to that, where is the speed/quality knee — how low
  can `m` go before the grid lands the wrong cell often enough that the fine tune
  can't recover it? (Measure kept-view/registration agreement vs `m`.)
- Supersampled grid (`m > 1`) as a cheaper sub-pixel approximation: how close does
  its discrete accuracy get to the continuous LK fine tune, and at what `m`/cost?
  This sizes *when* the grid is good enough vs. when the high-accuracy fine tune is
  worth it — the two coexist, this isn't grid-replaces-fine-tune.
