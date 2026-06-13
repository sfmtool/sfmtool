# Patch-Normal Refinement: Performance vs Benefit — 2026-06-13

Quantitative study of the v1 photometric patch-normal refinement
(`crates/sfmtool-core/src/patch_normal_refine.rs`, spec
`specs/core/patch-normal-refinement.md`): where the time goes, what each
tunable buys per millisecond, and which of the spec's proposed optimizations
are worth building. Measured on a 4-core Linux container against the four
checked-in datasets, reconstructed with the current `init_dataset_*.sh`
pipelines:

| dataset | images | image size | camera | points | mean views/patch |
|---|--:|---|---|--:|--:|
| seoul   | 17 | 270×480   | pinhole-ish | 1,115  | 3.5 |
| seattle | 26 | 360×640   | pinhole-ish | 3,331  | 4.4 |
| kerry   | 48 | 480×480   | **fisheye rig** | 889 | 3.9 |
| dino    | 85 | 2040×1536 | pinhole-ish | 19,034 | 4.8 |

Tooling added on this branch (all committed):

- `SFMTOOL_PROFILE=1` phase timers in the refinement hot path
  (`patch_normal_refine/prof.rs`) — atomic per-phase counters, a stderr
  summary per `refine_patch_cloud` batch, a single cached-flag branch when
  off. Profiling skews wall time by ~2% (checked seoul base with/without).
- `cargo bench -p sfmtool-core --bench patch_render` — criterion bench of the
  per-candidate render primitives at R = 8..32.
- `scripts/bench_normal_refine.py` — runs `PatchCloud.refine_normals` over a
  seeded 300-point subset per dataset under one-at-a-time parameter sweeps;
  reports wall time net of fixed overhead and quality metrics.

**Quality metrics and their limits.** `Φ` (the consensus photoconsistency) is
only comparable across runs that share the scoring config (resolution /
window / sampler / objective). The cross-config accuracy proxy is angular
agreement with a high-fidelity reference run (default R=32, steps 9, levels 4,
anisotropic, robust 3; saturation checks used an R=48 reference): `med°` =
median angle between a run's refined normals and the reference's, `≤5°` = the
fraction within 5°. There is no ground truth here — the reference is just the
most thorough run — and a reference shares *some* knob value with every sweep
point, so agreement numbers mildly favor configs close to the reference
(quantified below where it matters). The refinement is deterministic; repeats
only de-noise wall time (±5–10% on this container).

---

## 1. Changes landed (behavior-preserving, measured)

1. **Sequential row loops for small warp/remap destinations**
   (`PAR_MIN_PIXELS = 2048` in `warp_map.rs` / `remap.rs`). The per-candidate
   primitives parallelized over the rows of an R×R grid *nested inside* the
   already-parallel per-patch loop; for R ≤ 32 the rayon scaffolding cost an
   order of magnitude more than the work. Bit-identical output.
   - Primitives (bench, idle pool, R=16): `from_patch` 50 µs → 3.8 µs,
     `compute_svd` 31 µs → 6.4 µs; `remap_bilinear` 21 µs → 14.6 µs.
   - Single-patch `refine_patch_normal` (V=6, defaults): 242 ms → 95 ms
     anisotropic, 184 ms → 32 ms bilinear.
   - Saturated batch (seoul, 100 points, 4 cores): wall 1.62 s → 1.13 s
     (**1.45×**); per-render thread-summed cost 82 µs → 50 µs.
2. **`remap_aniso` skips the hi-level taps when their blend weight is zero**
   (`sigma_minor ≤ 1` ⇒ `frac == 0` exactly — the grazing-view case
   stretched along one axis but not minified). Bit-identical. Bench patch at
   R=24: 167 µs → 113 µs; end-to-end ~3% on seoul, in the noise on dino
   (where `sigma_minor > 1` dominates).

No algorithmic or default-value changes were made (§4: the current defaults
hold up).

---

## 2. Where the time goes

Phase shares of thread-summed CPU during `refine_patch_cloud` (base config:
R=16, anisotropic, robust 3, steps 7, levels 3; 300-point subsets; after the
§1 fixes). `ms/pt` is wall time per scored patch on 4 cores, net of the
per-call overhead.

| dataset | ms/pt | warp build | warp SVD | **remap** | gather+znorm | consensus (IRLS) | confidence stencil | masks | other |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| seoul   | 5.0  | 8.6% / 5.0 µs | 10.9% / 6.3 µs | **61.4% / 35.6 µs** | 6.1% | 9.7% / 19.6 µs | 9.1% | 0.3% | ~4% |
| seattle | 8.2  | 6.7% / 5.0 µs | 8.2% / 6.1 µs  | **71.1% / 53.1 µs** | 4.5% | 7.1% / 23.7 µs | 9.1% | 0.3% | ~3% |
| kerry   | 6.0  | 18.3% / 12.1 µs | 10.8% / 7.2 µs | **52.0% / 34.6 µs** | 5.8% | 9.5% / 24.3 µs | 8.5% | 0.4% | ~5% |
| dino    | 10.7 | 5.6% / 5.1 µs | 6.9% / 6.2 µs  | **75.2% / 67.4 µs** | 4.0% | 6.1% / 26.2 µs | 9.0% | 0.3% | ~3% |

Same workload with `sampler="bilinear"` (no SVD, single-tap level-0 sampling):

| dataset | ms/pt | warp build | remap | gather+znorm | consensus | confidence |
|---|--:|--:|--:|--:|--:|--:|
| seoul   | 2.6 | 16.5% | 48.6% / 15.1 µs | 11.4% | 18.5% | 9.0% |
| seattle | 3.2 | 16.6% | 49.2% / 15.1 µs | 11.1% | 17.6% | 9.1% |
| kerry   | 3.7 | 29.9% | 40.3% / 14.7 µs | 9.3%  | 15.1% | 8.7% |
| dino    | 3.6 | 15.7% | 48.5% / 16.0 µs | 10.7% | 17.0% | 8.6% |

Reading of the numbers:

- **Yes — with the anisotropic sampler, the multi-tap sampling is the
  bottleneck**: 52–75% of CPU, and it grows with image resolution (the warp's
  σ_major sets the tap count: dino's large footprints cost 67 µs per 16×16
  render, vs a flat ~15 µs for bilinear). The SVD that feeds it is small by
  comparison (7–11%; 6.2 µs ≈ one closed-form 2×2 SVD per pixel).
  End-to-end, anisotropic costs **1.6× (kerry) to 3.0× (dino)** over
  bilinear; at R=24 on dino the remap share rises to 77% (158 µs/render).
- The cost model is simple: `evals/pt × V × render`. Base config ≈ 99 Φ-evals
  per patch (3 levels × 29 disk-clamped candidates, + the final pass and the
  9-eval confidence stencil), ~3.5–4.8 renders per eval.
- **Fisheye (kerry) doubles the projection cost** (`from_patch` 12.1 µs vs
  5.0 µs — `ray_to_pixel` through the fisheye model), lifting warp build to
  18% (30% under bilinear). Still second to the sampling.
- The per-level **validity masks are negligible** (0.3–0.4%) and the
  per-candidate **support-reject rate is ~0.7%** of evals (e.g. 594 rejects /
  41k evals on the seoul reference run) — the existing once-per-level
  back-face cull already catches the bulk, so the spec's "cull before
  render" item has nothing left to win on these datasets.
- The **confidence stencil is a fixed ~9%** (9 extra Φ-evals per patch).
  IRLS is cheap at these view counts: robust 3 costs ~6–10% (≈ 8 µs per
  iteration per eval); `objective="mean"` drops that to ~1.3% but loses
  measurable quality (§3).
- Fixed per-call overhead (numpy conversion + pyramid build for *all* images,
  even when `point_ids` selects a few): seoul 0.01 s, seattle 0.04 s, kerry
  0.06 s, **dino 1.45 s** — dominates small-subset calls on large datasets.

---

## 3. Parameter perf-vs-benefit

One-at-a-time sweeps around the base config (R=16, steps 7, levels 3,
anisotropic, robust 3), 300-point seeded subsets. `med°` / `≤5°` = agreement
with the R=32/steps9/levels4/aniso/robust3 reference (see caveat in the
header). Cost is ms per scored patch (4-core wall).

### `resolution` (the dominant fidelity *and* cost axis)

| R | seoul ms/pt · med° · ≤5° | seattle | kerry | dino |
|---|---|---|---|---|
| 8  | 1.8 · 12.1° · 0.20 | 2.1 · 19.6° · 0.08 | 2.0 · 18.8° · 0.15 | 2.1 · 12.1° · 0.22 |
| 12 | 3.5 · 6.6° · 0.34  | 4.9 · 14.9° · 0.13 | 3.9 · 13.5° · 0.27 | 5.6 · 7.6° · 0.33 |
| 16 | 5.0 · 5.2° · 0.49  | 8.2 · 10.1° · 0.23 | 6.0 · 9.4° · 0.27  | 10.7 · 5.0° · 0.50 |
| 24 | 9.1 · 4.0° · 0.62  | 14.0 · 7.1° · 0.34 | 11.2 · 5.8° · 0.45 | 24.4 · 3.5° · 0.64 |
| 32 | 14.4 · 3.1° · 0.76 | 21.7 · 3.3° · 0.67 | 18.5 · 3.5° · 0.67 | 38.4 · 2.6° · 0.83 |

Cost grows ~R^1.8; agreement improves monotonically. Against an **R=48
reference** (removing the R=32 self-affinity): seoul 16/24/32 → 4.8°/4.0°/3.4°,
dino → 5.8°/3.6°/3.5°, kerry → 9.6°/5.6°/5.0°. So **dino saturates at R=24**
(24→32 buys 0.1°), seoul nearly so, while **kerry keeps improving past 24**
(weakly-constrained fisheye tracks scatter more — kerry's ≤2° fraction is
~0.05 at every R). The residual ~3–3.5° median against any deeper reference is
the grid search's intrinsic scatter, not a resolution effect.

### `init_steps` (grid resolution per level)

| steps | seoul | seattle | kerry | dino |
|---|---|---|---|---|
| 3 | 1.4 · 17.5° · 0.05 | 2.2 · 22.4° · 0.05 | 1.6 · 29.5° · 0.03 | 3.0 · 12.1° · 0.11 |
| 5 | 2.6 · 8.1° · 0.32  | 4.2 · 11.8° · 0.12 | 3.1 · 11.6° · 0.12 | 5.3 · 7.2° · 0.34 |
| 7 | 5.0 · 5.2° · 0.49  | 8.2 · 10.1° · 0.23 | 6.0 · 9.4° · 0.27  | 10.7 · 5.0° · 0.50 |
| 9 | 8.1 · 4.2° · 0.56  | 13.4 · 11.0° · 0.26 | 9.5 · 10.9° · 0.32 | 17.6 · 4.9° · 0.51 |

Evals/pt: 26 / 50 / 99 / 159. `steps=3` is **broken by construction**: the
per-level cone shrink is `2/(steps−1)` = 1.0, so the cone never contracts and
the search recenters blindly (kerry's median normal moved 50° from init).
`steps=5` is 2× cheaper than 7 but consistently 2–3° worse. `steps=9` costs
1.6× and buys ≤0.4° except on seoul (1°). **7 is the right default.**

### `refine_levels` (coarse-to-fine passes)

| levels | seoul | seattle | kerry | dino |
|---|---|---|---|---|
| 1 | 2.0 · 8.1° · 0.17 | 3.3 · 11.5° · 0.06 | 2.6 · 11.1° · 0.07 | 4.5 · 7.9° · 0.20 |
| 2 | 3.5 · 4.8° · 0.51 | 5.8 · 10.6° · 0.25 | 4.3 · 10.3° · 0.30 | 7.6 · 5.1° · 0.48 |
| 3 | 5.0 · 5.2° · 0.49 | 8.2 · 10.1° · 0.23 | 6.0 · 9.4° · 0.27  | 10.7 · 5.0° · 0.50 |
| 4 | 6.5 · 5.1° · 0.49 | 10.5 · 10.2° · 0.22 | 7.6 · 9.6° · 0.25 | 13.6 · 5.2° · 0.49 |

1 level is clearly insufficient; **2 and 3 tie on agreement** (2 even noses
ahead on three datasets, within scatter) while 3 reaches slightly higher Φ
(mean ΔΦ +0.001–0.005 — it is never worse than its own init by construction).
4 levels is pure waste (final grid spacing is already ~0.3°). Keep 3 as the
thorough default; **2 is the documented economy point (×0.7 cost)**.

### `sampler` (the anisotropic question)

| sampler | seoul | seattle | kerry | dino |
|---|---|---|---|---|
| anisotropic | 5.0 · 5.2° · 0.49 | 8.2 · 10.1° · 0.23 | 6.0 · 9.4° · 0.27 | 10.7 · 5.0° · 0.50 |
| bilinear    | 2.6 · 5.4° · 0.44 | 3.2 · 10.0° · 0.20 | 3.7 · 11.8° · 0.24 | 3.6 · 5.8° · 0.47 |

**The sampler barely moves the found normals.** On the pinhole datasets the
bilinear search lands within scatter of the anisotropic one (±0.2–0.8°
median, sign varies) — including dino, whose large footprints make bilinear
alias hardest. Cross-checking against a *bilinear* R=32 reference on dino
gives the mirror image (aniso 5.9°, bilinear 5.4°), i.e. the two samplers'
high-fidelity answers coincide and neither search is systematically off.
Kerry is the exception with a small but reproducible anisotropic edge
(9.4–9.6° vs 10.1–11.8°). What the sampler *does* change is the **reported
Φ** — bilinear-scored Φ is depressed by aliasing noise (dino mean Φ 0.78 vs
0.86), which matters because `photoconsistency` and the Φ-curvature
`confidence` are consumer-facing outputs — and the **cost: 1.6× (kerry) to
3.0× (dino)**.

### `objective` / `robust_iters`

| config | seoul | seattle | kerry | dino |
|---|---|---|---|---|
| mean (= robust 0) | 4.4 · 6.4° · 0.39 | 7.5 · 10.3° · 0.22 | 5.1 · 13.6° · 0.21 | 10.0 · 6.6° · 0.39 |
| robust 1 | 4.7 · 5.9° · 0.44 | 7.8 · 10.1° · 0.23 | 5.4 · 13.2° · 0.23 | 11.5 · 5.7° · 0.46 |
| robust 3 | 5.0 · 5.2° · 0.49 | 8.2 · 10.1° · 0.23 | 6.0 · 9.4° · 0.27  | 10.7 · 5.0° · 0.50 |
| robust 5 | 5.3 · 6.0° · 0.41 | 8.6 · 12.4° · 0.19 | 6.6 · 12.2° · 0.25 | 12.1 · 6.0° · 0.45 |

Robust weighting is nearly free (+5–15% total) and buys real convergence on
the occlusion-prone datasets (kerry 13.6°→9.4°, dino 6.6°→5.0°); 5 iterations
is consistently *worse* than 3 (over-concentrated weights; it also drops more
patches through the effective-view gate — kerry scores 202 vs 229). The mean
objective scores more patches (no effective-view gate: kerry 266) but
converges worse. **`RobustWeighted { iters: 3 }` is the right default.**

## 4. Recommended defaults

The v1 defaults survive contact with the data; none of the measured knobs is
mis-provisioned enough to justify a change in this branch:

| knob | current (core / binding / script) | verdict |
|---|---|---|
| `resolution` | — / 24 / 32 (`--patch`) | **Keep 24** in the binding: it is the knee on dino/seoul and near-knee on kerry. 16 is the right *fast* setting (½ the cost, ~1–4° looser); 32 only pays on distorted/weak data (kerry). The crossval script's 32 is its render size doing double duty — fine for a thorough offline tool. |
| `init_steps` | 7 | **Keep.** 5 loses 2–3°; 9 costs 1.6× for ≤0.4–1°. Guard against `steps=3` (no cone shrink — consider a `max(4)` or doc note). |
| `refine_levels` | 3 | **Keep**, marginally over-provisioned: 2 ties on agreement at ×0.7 cost; 4 is waste. |
| `sampler` | `Anisotropic` | **Keep**, for output fidelity (unbiased Φ and confidence) and the kerry edge — but document `bilinear` as a 1.6–3× cheaper search whose *normals* are equivalent on low-distortion data; and see opportunity #2 for keeping the fidelity at near-bilinear cost. _Status (2026-06-13): superseded — the default was changed to `Bilinear` (speed prioritized; the found normals are equivalent on the pinhole datasets). `Anisotropic` stays an opt-in for unbiased Φ/confidence, and opportunity #2 (single-level pyramid sampling) remains the way to recover its fidelity cheaply._ |
| `objective` | `RobustWeighted { iters: 3 }` | **Keep.** Demonstrably better than `mean` and than 1 or 5 iterations, at ~6–10% of runtime. |
| `angular_range_deg` | 25 | Not swept (the init normals here genuinely sit ~25–30° off the optimum, so narrowing would mostly truncate the search); revisit together with the GN polish. |

Distorted-rig note: kerry wants *more* fidelity, not different structure —
higher R helps it more and saturates later, anisotropic sampling has a real
(small) edge, and robust weighting matters most there. A caller refining
fisheye rigs should prefer R=24–32 + anisotropic; the defaults already point
that way.

## 5. Opportunities, prioritized

Grounded in §2's shares; "expected" speedups are for the base config on these
datasets (4 cores), multiplicative where independent.

1. **[landed] Sequential small-grid dispatch** — 1.45× saturated batch wall
   (seoul), 2.5–5.8× on isolated calls; see §1.
2. **Fused masked f32 render path** (spec item 4, first bullet) — one pass
   per view that projects *only the frozen-support pixels* (the GaussianDisk
   mask is ≤ π/4 of the square grid before validity), samples straight into a
   per-thread f32 scratch, and skips the `WarpMap` + `ImageU8`
   intermediates. Eliminates ~21%+ of all sampling work (non-mask pixels),
   the u8 round-trip (which currently quantizes *before* z-normalization —
   a fidelity win for Φ and the future GN gradients), and per-candidate
   allocations. With remap+warp+znorm at 75–85% of CPU, expect **~1.3–1.6×**
   end-to-end. Medium change (new render entry point + `normalized_stack`
   rework); the natural next step.
3. **Fidelity schedule** (spec item 4, second bullet) — coarse levels only
   rank basins: run levels 1..n−1 at R=8/bilinear (measured ~1.2–2 ms/pt) and
   reserve full R + anisotropic for the last level and final pass. With ~⅔ of
   evals in coarse levels, projected dino cost at R=24 final fidelity:
   ~24.4 → ~9–10 ms/pt (**~2.5×**), stacking with #2. Needs a top-k coarse
   carry-over to guard mis-ranking (the R=8 column's 12–19° shows coarse
   ranking alone is not trustworthy for the *final* answer, which is exactly
   why only the basin choice may be delegated to it). Medium change; verify
   with the agreement metric (target: within scatter of the unscheduled run).
4. **Cheap anti-aliasing: single-level pyramid sampling** — answer to "what's
   the cheapest way to keep aniso's benefit": since the *normals* are already
   sampler-insensitive, the residual value of `remap_aniso` is unbiased Φ /
   confidence and the kerry edge. A sampler that picks one pyramid level from
   the (per-level frozen) σ and takes a single bilinear tap (or fixed 2-tap
   trilinear) would cost ~bilinear+ε instead of 35–158 µs — capturing the
   minification de-aliasing and giving up only the directional filtering.
   Expect anisotropic-like Φ at **~2–3× less remap cost** on dino-like data.
   Small-medium change; measure Φ-bias and kerry agreement before adopting as
   default. (Related micro-lever, unmeasured: `MAX_ANISOTROPY` 16 → 8 caps
   the tap tail.)
5. **Freeze the warp SVD per grid level** — `compute_svd` runs per candidate
   per view (7–11% of CPU) but the Jacobian varies little
   across a level's candidates; the mask is already frozen per level, so
   freezing the σ/direction field at the level center is consistent with the
   existing approximation. **~1.1×**, small change, approximation risk ~nil.
6. **Confidence on demand** — the 9-eval stencil is a constant ~9% even when
   the caller ignores `confidence` (e.g. parameter sweeps, quick passes). A
   `compute_confidence: bool` (default true) is a trivial **1.1×** for those
   callers.
7. **Subset-aware pyramid build in the binding** — `refine_normals` builds
   pyramids for *all* images (dino: 1.45 s/call) even when `point_ids`
   touches a handful of views. Building only the views referenced by the
   selected patches makes small-subset calls on large datasets ~free. Small
   Python-visible change, no effect on full-cloud runs.
8. **Stochastic view subsets** (spec item 3) — not for these datasets:
   V̄ = 3.5–4.8 with `min_views = 3` leaves no room. Becomes the dominant
   lever only for dense captures (V ≳ 10). Defer.
9. **Per-candidate cull-before-render** (spec item 2) — already effectively
   done: the per-level back-face cull plus frozen-support reject leaves a
   ~0.7% candidate reject rate. **No measurable win left**; skip.
10. **GPU stage 1** (spec item 4) — the search is textured-quad sampling
    (pyramids ≈ mipmaps, `remap_aniso` ≈ hardware anisotropic filtering);
    the CPU path after #2–#5 will still be ~10–30 s for a dino-scale cloud,
    so this stays the order-of-magnitude follow-up for million-point clouds.

Cloud-scale projection (full dino, 19k points × 87% scored, 4 cores):
current binding defaults (R=24, aniso) ≈ **7 min**; with #2+#3 ≈ **2–3 min**;
bilinear R=16 today ≈ 60 s. seoul/seattle/kerry full clouds are already
7–45 s at binding defaults.

## 6. Caveats

- Single 4-core container; wall numbers carry ±5–10% noise (repeats where
  cheap; phase *shares* were stable across runs). Thread-summed CPU ≠ wall.
- The reference runs are thorough, not ground truth; agreement medians
  conflate true error with the search's ~3° grid scatter. The
  confidence-gated agreement metric (`med_angle_ref_hiconf_deg`) did not
  separate cleanly from the raw median on these datasets.
- Φ comparisons across scoring configs (resolution / sampler / window /
  objective) are not meaningful and were not used; within-config ΔΦ and
  cross-config angular agreement were.
- 5–34% of sampled points (seattle 5%, dino 13%, kerry 24%, seoul 34%) fail
  the validity gates (min 3 views, valid-fraction, effective-view floor) and
  return unrefined; ms/pt is normalized by *scored* patches, and
  gate-failing patches cost ~5 µs each.
- The sweeps ran before the §1.2 `frac == 0` remap skip landed; its effect on
  the swept timings is ≤3% (re-verified base configs bit-identical and within
  noise).
