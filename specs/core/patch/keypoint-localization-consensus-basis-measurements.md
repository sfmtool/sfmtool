# Keypoint Localization Consensus-Basis Measurements

This file records the measurements behind the default of the consensus-basis
cap in [keypoint-localization-consensus-basis.md](keypoint-localization-consensus-basis.md#why-the-default-is-eight-views).
The cap congeals at most `K` of a point's views into a consensus template and
registers every other view once against it. The runs compare `K = 0`
(uncapped) with `K = 8, 12, 16` and the `basis_force_track_views` and
`basis_pick` variants on one high-`V` capture: first the localizer's own cost
and per-observation agreement, then the downstream size cull and bundle
adjustment.

## Arms and metrics

Arms on a high-`V` reconstruction (expanded view sets reaching `V ≥ 100`) and a
moderate-`V` control (typical `V ≈ 15–40`):

| arm | question |
| --- | --- |
| `K=0` | baseline (bit-identity check for `V ≤ K` points included) |
| `K=12`, force-track, `TopScore` | the default proposed before these runs |
| `K=12`, no force-track | do track views earn reserved seats? |
| `K=12`, `Strided` | does duplicate-frame clustering in the top scores hurt the tail fit? |
| `K=8`, `K=16` at the winner | sensitivity of the cap |

The moderate-`V` control has **not** been run; every number below is from the
high-`V` capture. The `seoul_bull` fixture in the test suite covers only the
`candidates ≤ K` no-op, which is a different question.

Metrics per arm, against `K=0`: localize wall time; per-observation keypoint Δ
(median / p99 source px); observation drop-set churn; tail-view ZNCC
distribution (a smeared or arc-biased basis template shows up as depressed tail
ZNCC on views far from the basis's viewpoints); and the downstream
embed→size-cull→BA+refine chain's reprojection / yield / rogue-normal metrics
for the candidate default.

## DnDTabletop, inside the localizer (2026-07-26)

`sfmr/cleanup/gt-clean-01-ba.sfmr`, 337 images / 132,965 points, `--patch-size
5`, `resolution 24`, default `PlusDescent`. `select_views` produces a mean of
51.7 views/point, p99 236, max 323 — the high-`V` case the cap targets.

**Cost** — `SFMTOOL_PROFILE=1` on a 12,000-point subset. Thread-summed CPU
seconds carry ±20 % run-to-run variance from memory-bandwidth contention on a
shared machine, so the exact work counters (which are deterministic) are the
reliable signal; `render px` is `Σ` cache area, `basis · (R+4m)² + tail ·
(R+2m)²`.

| arm | `localize_total` | `loo_gram` + `loo_template` | `render_context` | `search_shift` | searches | render px |
| --- | --- | --- | --- | --- | --- | --- |
| `K=0` | 1199.0 s | 519.8 s (43.3 %) | 435.7 s | 189.7 s | 2,130,550 | 1.410 G |
| `K=8` | 436.1 s | 7.3 s (1.7 %) | 303.6 s | 113.5 s | 871,226 | 0.888 G |
| `K=12` | 440.0 s | 13.7 s (3.1 %) | 298.4 s | 113.6 s | 989,488 | 0.931 G |
| `K=16` | 360.1 s | 18.2 s (5.1 %) | 238.1 s | 91.0 s | 1,088,657 | 0.969 G |
| `K=12`, no force-track | 450.6 s | 14.6 s | 304.1 s | 116.0 s | 992,500 | 0.931 G |
| `K=12`, `Strided` | 467.3 s | 14.2 s | 316.3 s | 120.4 s | 969,964 | 0.931 G |

The quadratic terms do what the cap is for: 43.3 % of the pass at `K=0`, 2–5 %
capped. What remains is per-view and barely depends on `K` — every view still
renders a cache (the cap only shrinks the tail's tile from `(R+4m)²` to
`(R+2m)²`, a 1.5× total-area cut) and still runs at least one search. So the
whole `K = 8…16` band lands within measurement noise of each other at ~2.6–3.3×
the `K=0` pass, and pushing `K` lower buys nothing further.

**End to end** — `sfm embed-patches --patch-size 5` over the full
reconstruction, `K=0` then `K=12`, back to back:

| stage | `K=0` | `K=12` |
| --- | --- | --- |
| round-1 normal refine | 105.0 s | 60.1 s |
| view selection | 58.1 s | 56.6 s |
| **keypoint localization** | **484.5 s** | **107.7 s** |
| round-1 sub-pixel refine | 235.6 s | 194.6 s |
| round-2 normal refine | 114.7 s | 86.7 s |
| round-2 sub-pixel refine | 214.9 s | 154.5 s |
| whole command | 1278 s | 699 s |
| points written | 126,737 | 128,431 |

Normal refinement and view selection do identical work in both arms, so their
gap (105.0 vs 60.1 s) is the machine-load difference between the two runs;
correcting the localize column by it puts the pass at ~2.6× rather than the raw
4.5×, matching the profiled figure, and the whole command at ~1.3–1.8×. The
capped arm also writes 1.3 % more points, the compaction's view of the +4.5 %
observations the localizer kept.

**Quality** — 40,000-point subset, per observation against the `K=0` arm,
matched on `(point, image)`. Churn is the symmetric difference of the kept
observation sets over `K=0`'s count. `zncc` columns are the reported per-view
ZNCC medians for basis and tail members.

| arm | observations | churn | Δ median | Δ p99 | Δ > 1 px | basis zncc | tail zncc (med / p10) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `K=0` | 1,307,430 | — | — | — | — | 0.952 | — |
| `K=8` | 1,366,334 | 29.6 % | 0.852 px | 3.42 px | 41.7 % | 0.972 | 0.919 / 0.781 |
| `K=12` | 1,366,677 | 27.5 % | 0.783 px | 3.31 px | 37.9 % | 0.970 | 0.918 / 0.781 |
| `K=16` | 1,367,075 | 25.6 % | 0.722 px | 3.22 px | 34.4 % | 0.968 | 0.918 / 0.782 |
| `K=12`, no force-track | 1,370,195 | 27.1 % | 0.770 px | 3.29 px | 37.0 % | 0.970 | 0.919 / 0.782 |
| `K=12`, `Strided` | 1,363,888 | 26.1 % | 0.734 px | 3.24 px | 35.0 % | 0.967 | 0.926 / 0.791 |

Readings:

- The cap is **not** a small perturbation. Half the observations move more than
  ~0.7 px and a quarter of the kept set churns; divergence from `K=0` shrinks
  monotonically as `K` grows, as it must. The observations `K=0` keeps and a
  capped arm drops score the same median ZNCC in `K=0` (0.952 vs 0.952) as the
  ones it keeps, so the churn is not a targeted cull of weak observations.
- Capped arms keep **more** observations (+4.5 %): a tail view faces the
  relative-ZNCC gate once against the finished template instead of surviving a
  multi-round cull whose consensus (and therefore whose bar) moves under it.
  The no-basis path is *not* what produces the surplus: `N_TAIL_NO_BASIS`
  counts 244 of 475,645 tail views (0.05 %) on this capture, so all but a
  rounding error of the extra observations are genuinely registered against a
  template.
- Tail ZNCC (median 0.918) sits below basis ZNCC (0.970) and below `K=0`'s
  all-view 0.952, but the three are not the same measurement: `K=0` scores each
  view against a leave-one-out consensus of ~50 views, the tail against a sharp
  12-view template, and a sharper reference scores lower for the same fit. The
  tail p10 (0.78) shows no collapsed lower tail — no evidence of a smeared or
  arc-biased template.
- `basis_force_track_views` is **not** load-bearing here: with and without it,
  every metric agrees to within 0.5 %. It is kept on for provenance, not for a
  measured gain.
- `Strided` is marginally the best of the `K=12` variants on divergence
  (26.1 % churn, 0.734 px) and tail ZNCC (0.926) — the top-score band does
  contain redundant near-duplicate frames — but the margin is inside the
  `K=12` → `K=16` gap, so raising `K` is the simpler lever.
- **Divergence from `K=0` is a measure of change, not of error** — the
  all-view consensus is the behaviour the cap exists to replace, not a ground
  truth. Read without that anchor, the internal metrics favour a *smaller*
  `K`: basis ZNCC rises monotonically as `K` shrinks (0.972 / 0.970 / 0.968
  for `K` = 8 / 12 / 16 — fewer, better-ranked members give a sharper
  template), tail ZNCC is flat across `K` (an 8-view template registers the
  tail as well as a 16-view one), and the exact work counters are lowest at
  `K=8`. A larger `K` buys only proximity to `K=0`. The choice between the
  small-`K` and large-`K` ends of the band is therefore delegated entirely to
  the downstream comparison below, on arms `K ∈ {0, 8, 16}`.

These localizer-internal metrics cannot by themselves justify a default; the
downstream embed→size-cull→BA+refine runs below are what the default rests on.

## DnDTabletop, downstream chain (2026-07-27)

Arms `K ∈ {0, 8, 16}` through embed → `--filter-by-patch-size 3.0` →
`--bundle-adjust --refine-normals --refine-keypoints` on the same high-`V`
capture (same build, sequential runs). Rogue % = normals > 70° off their
8-NN consensus.

| arm | embed wall | pts | obs | reproj med / p90 | rogue % |
| --- | --- | --- | --- | --- | --- |
| `K=0` | 21m 0s | 117,428 | 3.986 M | 1.308 / 1.831 | 3.99 |
| `K=8` | 10m 22s | 119,411 | 4.180 M | 1.372 / 1.947 | 4.31 |
| `K=16` | 11m 8s | 118,771 | 4.181 M | 1.357 / 1.926 | 4.13 |

Restricted to the 115,777 points common to all three arms (removes the
yield-composition confound between arms):

| arm | reproj med / p90 | rogue % | obs/pt |
| --- | --- | --- | --- |
| `K=0` | 1.307 / 1.826 | 3.96 | 34.02 |
| `K=8` | 1.365 / 1.929 | 4.25 | 35.20 |
| `K=16` | 1.351 / 1.909 | 4.08 | 35.28 |

The capped arms' extra-only points (kept by them, culled by `K=0`): median
reproj 1.76 px, rogue 6.4–6.9 % — marginal but usable observations, not junk.

Readings:

- The cap halves end-to-end embed wall and raises yield ~1.7 % pts / ~4.9 %
  obs, at a small error cost that persists on the common subset: +3–4 %
  median reproj and +0.1–0.3 pp rogue. Even on common points the capped arms
  carry ~1.2 more obs/pt (the single-shot tail gate keeps marginal
  observations the moving multi-round bar culls), so the residual deltas
  conflate keypoint quality with observation composition; separating them
  needs a tail-bar-tightening arm that matches `K=0`'s obs/pt.
- Between the capped arms, `K=16` beats `K=8` on every downstream metric
  (small margins) — the sharper-template internal reading of small `K` did
  not hold downstream on this capture.
- The default is **`K=8`** at every layer — the small-`K` end of the validated
  band, taking the halved wall and the extra yield at the measured +3–4 %
  median-reproj cost. The `K=8` / `K=16`
  downstream margins (1.365 vs 1.351 med, 4.25 vs 4.08 % rogue on the common
  subset) are small; `K=8` congeals the sharpest template and does the least
  work. `K=0` (`--localize-basis-views 0`) remains the choice where error
  metrics are the product — e.g. ground-truth cleanup ladders. Still
  outstanding: the tail-bar experiment separating gate composition from
  keypoint quality, and the moderate-`V` control.
