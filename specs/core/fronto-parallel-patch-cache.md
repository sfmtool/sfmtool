# Fronto-Parallel Patch Cache for Normal Refinement

**Status:** Phases 1 and 2 implemented, and **the cache is the default**
(`CacheMode::FrontoParallel`, `cache_supersample = 2` — the quality-preserving
operating point; `cache=off` / `quality=fine` opts back into the exact
source-rendering path). The cache lives in
`sfmtool-core/src/patch_normal_refine/fronto_cache.rs`, selected by
`NormalRefineParams { cache: CacheMode, cache_supersample }`, exposed through
`PatchCloud.refine_normals(cache=…, cache_supersample=…)` and
`sfm xform --refine-normals cache=…/quality=…`. The resample is runtime-dispatched
to an AVX2 kernel (Phase 2) with the scalar path as reference/fallback. Builds on
`specs/core/patch-normal-refinement.md` (the search this accelerates) and
`specs/core/patch-cloud.md` (`OrientedPatch`, `WarpMap::from_patch`, `remap_*`).
Prototype exploration and measurements: `reports/2026-06-15-patch-cache-status.md`.

## Problem

Refinement (`coarse_to_fine` in `patch_normal_refine.rs`) scores each candidate
normal by **re-rendering** the patch into every observing view —
`WarpMap::from_patch(n') + remap_bilinear` off the source pyramid — and the
per-candidate render is ~80% of CPU (`reports/2026-06-13-perf-patch-normal-refinement.md`).
A coarse-to-fine search with `init_steps=7`, `refine_levels=3` evaluates on the
order of 3 levels × ~37 candidates × `V` views renders per patch. The render is
the cost; everything that removes redundant rendering compounds.

## Idea

Within a refinement the patch **center, size, and the 3D point are fixed** — only
the normal's 2 DOF move. So the texture seen by each view is fixed up to a planar
re-warp. **Render one base patch per view, once, and warp it for every candidate**
instead of re-rendering from the source.

The fronto-parallel variant makes two specific choices:

1. **Render the base fronto-parallel to each camera, once, up front.** For view
   `v`, build a patch whose plane normal points from the 3D point at the camera
   centre `Cᵥ`, render it (supersampled) from the source pyramid, and keep it for
   the whole refinement. The source image is touched **exactly once per view**.
2. **Warp every candidate from that base by an affine map.** Treating the small
   patch as a differential, the patch→patch correspondence is an affine
   (orthographic) map, not a full homography. Every candidate at every level is
   an affine resample of the cached base — **the source image is never read
   again**.

## Geometry and the maps

A patch carries a centre `X`, in-plane axes `(u, v)`, and half-extents; its
`R×R` grid samples `(s, t) ∈ [-1, 1]²` → world via `OrientedPatch::to_world`.

For each view we fit a **grid → undistorted-normalized image** map from the four
projected grid corners. "Undistorted-normalized" means the pinhole part
`(x/z, y/z)` of the corner in camera space — the lens distortion is *omitted*.
This is the key to projection-independence: distortion is the same fixed image
warp applied to base and candidate, so it **cancels** in the base↔candidate
correspondence; the map is exact for *any* camera model (pinhole, distorted,
fisheye) as long as both maps use the same undistorted corners.

- Base (fronto): `A₀` = affine fit of the fronto grid corners → undistorted-norm.
- Candidate `n'`: `A'` = affine fit of the candidate grid corners → undistorted-norm.
- Candidate-grid → base-grid map: `φ = A₀⁻¹ · A'`.

`A₀` and `A'` are affine (homogeneous 3×3 with last row `[0,0,1]`), so `φ` is
affine: the projective denominator is a constant `1`. The kernel resamples the
cached base at `φ·(col, row, 1)` with bilinear taps — no per-pixel divide.

The affine fit is the exact 3-corner affine (`(0,0)`, `(R-1,0)`, `(0,R-1)`); no
8×8 DLT solve, unlike the full-homography variant.

## Design choices (and why)

- **Fronto-parallel orientation.** The base that faces the camera is the
  *least-foreshortened* render, so it carries the most source resolution to
  resample candidates from, and a tilted candidate's footprint is generally
  *smaller* than the fronto footprint — coverage is mostly automatic (the
  `base_margin` knob measured ~0 effect on accuracy, confirming coverage is not
  the limiter). The alternative — basing on the current level-centre normal — is
  what the homography cache does; it must re-render per level because the centre
  moves, losing the single-render win.

- **One render up front, not per level.** This is the speed win. The
  fronto base is candidate-*and* level-independent (it depends only on geometry),
  so the three per-level renders of the homography cache collapse to one. Measured
  end-to-end this is the difference between ~1.9× and ~2.25×.

- **Affine (differential) map instead of homography.** Two payoffs: a cheap
  3-corner affine fit (no DLT) and a divide-free kernel. The cost is accuracy in
  the **tail** — an affine is the first-order approximation of the patch
  homography, slightly softer at large tilt, which flips a few flat-Φ/ambiguous
  candidates (p90 18° vs the homography's 15°; Φ −0.0101 vs −0.0074). Median and
  high-confidence accuracy are unchanged (both 2.07°), so for the common case the
  approximation is free.

- **Undistorted-normalized corners.** Makes the cache exact for distorted and
  fisheye rigs with no special-casing (measured ~0 effect to add/remove, i.e. it
  is *already* correct, which is the point — distortion cancels).

- **Supersample the base.** Rendering the base denser than the candidate grid
  (`set_patch_cache_supersample`) sharpens the resample and is the **only** lever
  that moves accuracy: it halves the median (ss1 median 3.9° → ss2 2.07°). It
  costs a bigger base render. This is the accuracy/speed knob.

- **Kernel layout: packed `u32`, planar output, masked support.** The kernel is
  **gather-bound**, so the wins are about *gathered bytes* and *store shape*, not
  arithmetic: pack the base RGB into one `u32`/pixel (one `vpgatherdd`/tap → 4
  gathers instead of 12); write each channel to its own plane (3 vector stores
  vs 24 scalar interleave-writes); resample only the GaussianDisk support pixels
  and write them straight into the scorer's layout (fuses the scorer's gather).
  Runtime-dispatched AVX2 (the `kdforest`/`is_x86_feature_detected!` pattern) with
  a scalar reference.

- **Branch-free kernel with a guarded base.** The map is affine, so the per-pixel
  float coordinate clamps can be dropped. But the "candidate shrinks inside the
  base" assumption **fails** when the surface normal is oblique to the camera (the
  candidate is tilted by the full obliquity + cone, and its sheared quad pokes a
  few pixels past the edge; a grazing/degenerate candidate escapes arbitrarily —
  a naive no-clamp gather panicked on real data). The safe form keeps a
  **replicate-padded** base (a small guard border) plus **one branch-free integer
  `min`/`max`** on the pixel coordinate, which also makes degenerate candidates
  memory-safe. It is 1.11× at the microbench but within noise end-to-end (the
  resample is a slice of the un-cacheable work), so it is a clean simplification,
  not an e2e mover.

## Results (dino, 85 img, 3000 pts, R=32, ss2)

| variant | speedup | median Δnormal | p90 Δ | hi-conf median | Φ vs baseline |
|---|---|---|---|---|---|
| baseline (re-render every candidate) | 1.0× | — | — | — | 0.7671 |
| per-level homography cache, min0 | 1.9× | 2.07° | 14.9° | 2.07° | −0.0074 |
| **fronto-parallel single render** | **~2.25×** | **2.07°** | 18.0° | **2.07°** | −0.0101 |

The fronto normals are **as good** (Φ-equivalent) on the median and the
high-confidence subset; the angular disagreement concentrates on low-confidence /
flat-Φ points where several normals are near-tied, i.e. it is ambiguity, not
error. Kernel microbench (32×32×3): masked perspective ≈5.1 µs → affine ≈5.0 µs
(divide dropped) → branch-free ≈4.6 µs; packed+planar is ~2.8× over the generic
runtime-R kernel.

## Limitations / when it does not apply

- **Flat-Φ data** (weak parallax / low texture): the affine softening costs a few
  degrees in the tail. Prefer no cache when the tail matters more than wall time.
- **Extreme tilt**: the differential/affine approximation degrades; the guard +
  int clamp keep it safe but the resample is approximate there.
- **Non-3-channel images**: the packed kernel assumes RGB; the base render bails
  (falls back to source rendering) otherwise.
- **Out-of-frame candidates are not rejected.** The source path drops a candidate
  whose frozen-support pixels leave any view's frame (so zero-fill can't fake
  cross-view agreement); the cache instead resamples the replicate-padded base
  *edge*. This is an argmax-affecting divergence, but measured Φ-equivalent in
  practice — the fronto base is the least-foreshortened render, so its support
  covers the candidate's footprint.
- **Per-patch view coverage is fixed at the seed.** Bases are rendered for the
  views front-facing at the search seed; a view that only becomes front-facing
  (and valid) at a *drifted* later-level centre has no base, which ends that
  patch's search one level early rather than dropping the point (level 0 — seeded
  at the search centre — is always covered).

## Implementation plan (production)

The prototype proves the algorithm; production needs it parameterized,
deterministic-first, tested, and exposed. Land it as two reviewable changes —
"changes the numbers" first, "changes only speed" second.

### Phase 1 — algorithm, scalar kernel, parameterized (the merge that matters)

> _Status (2026-06-15): done. `patch_normal_refine/fronto_cache.rs`
> (`FrontoCache`, `prerender`, `eval_phi`, scalar packed/planar/masked-support
> resample with the guarded base + single int clamp); `CacheMode` +
> `cache_supersample` on `NormalRefineParams` (default `Off`); the
> `PatchCloud.refine_normals(cache=, cache_supersample=)` binding; the
> `--refine-normals cache=/cache_supersample=/quality=` CLI keys and the
> coarse/fine preset; unit tests for the affine fit and padding plus
> `tests/test_patch_normal_refine.py` (Φ-equivalence + population on seoul_bull,
> distortion-independence on the kerry_park fisheye rig) and
> `tests/xform/test_refine_normals.py` (preset + validation). Measured ~2.3× at
> Φ-equivalent median on dino R=32. Phase 2 (AVX2) pending._

1. **Lift the cache into the refine module**: the fronto base render
   (`prerender`), the affine fit and composition, and `eval_phi`. (Written fresh
   as production code rather than lifted from the prototype branch, which never
   reached `main`.)
2. **Parameterize via `NormalRefineParams`**, not globals: add
   `cache: CacheMode { Off, FrontoParallel }` (room to grow) and
   `cache_supersample: f64`. Default `Off` so existing behavior is unchanged
   until opted in.
3. **Scalar resample only.** Portable, deterministic, no `target_feature`. The
   scalar path *defines* the production result so the SIMD follow-up is a pure
   speed change. Keep the packed-`u32` + planar + masked-support *layout* (it is
   the data-movement win and is independent of SIMD).
4. **Wire the coarse/fine quality preset** in `sfm xform --refine-normals`
   (`specs/cli/xform-refine-normals-command.md`): a `quality=coarse` selects
   `cache=fronto, cache_supersample=2` (the ~2.25× / Φ-equivalent-median point);
   `quality=fine` selects `cache=off`. Add `cache` / `cache_supersample` to that
   command's parameter table.
5. **Tests** (`tests/rust_bindings/` + a core unit test):
   - cache-on vs cache-off on a fixture (e.g. `seoul_bull`) — assert mean Φ within
     a tolerance and the **scored-point count is unchanged** (the cache must not
     drop points);
   - the affine composition round-trips a known plane;
   - distortion independence: a distorted/fisheye view (kerry) refines without the
     cache diverging from source rendering beyond tolerance.
6. **Update specs/docs**: promote this file's Status to "v1 implemented", update
   the refinement spec and the CLI command spec.

### Phase 2 — AVX2 kernel (pure speed, no behavior change)

> _Status (2026-06-15): done. `resample_support` runtime-dispatches (the
> `kdforest` `is_x86_feature_detected!` pattern) to `resample_support_avx2`
> (packed `vpgatherdd` — 4 gathers/tap not 12 — branch-free guarded base, planar
> stores), with `resample_support_scalar` the reference and the `n % 8` tail.
> Correctness test `resample_avx2_matches_scalar` (incl. an off-base-edge map);
> the kernel is **3.5×** the scalar reference at the microbench
> (`resample_bench`, `--ignored`). End-to-end the cache is ~2.3–2.5× off vs on at
> unchanged accuracy (Φ and normals identical to Phase 1 within rounding); the
> kernel win is gather/Amdahl-capped by the un-cacheable work and the
> per-candidate affine inverse, as the report predicted._

1. Add the runtime-dispatched AVX2 masked affine kernel (packed/planar,
   branch-free with the guarded base) behind the same path, with the **scalar
   kernel as the reference**.
2. Correctness test: `avx2 ≈ scalar` within an eps over a fixture's candidates;
   an on-demand `resample_bench` (`#[ignore]`) is the microbenchmark.
3. Because the scalar path already fixed the numbers, this merge cannot regress
   quality — it only needs to prove "same output, faster."

### Out of scope (follow-ups)

- The un-cacheable e2e work (level-0 base render, consensus, final pass,
  confidence) is the remaining Amdahl ceiling; attack it separately.
