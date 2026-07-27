# Keypoint localization — consensus-basis cap (basis congealing + tail registration)

> Status: **implemented (2026-07-26)** —
> `crates/sfmtool-core/src/patch/keypoint_localize.rs` (orchestration) and
> `keypoint_localize/basis.rs` (the ranking pick), exposed as
> `PatchCloud.localize_keypoints(basis_max_views=…)`,
> `sfm embed-patches --localize-basis-views` and
> `sfm xform --localize-keypoints basis_max_views=…`. The Rust-layer **and**
> pipeline defaults are `0` = off (bit-identical current behavior); flipping the
> pipeline default is a separate decision from the A/B validation below.

## Motivation

Keypoint localization congeals **every** view of a point's view set against a
leave-one-out consensus of all the others. Per congealing round that is
`O(V²·n)` work (`V` views, `n` support pixels): one `V×V` Gram build plus, per
view, a Gram-space IRLS and a pixel-space template materialization over the
other `V−1` views. The expanded view sets produced by `select_views` are
unbounded — a point covisible across a long capture can carry `V` in the
hundreds — so the quadratic terms dominate the pass, concentrated in the
high-`V` tail.

The statistics don't want those views in the consensus either. The consensus
template's noise floor is reached after a modest number of well-matched views;
further views contribute redundant appearance while widening the warp-error
spread the robust reweighting must absorb (each view is rendered through an
imperfect patch frame, so a consensus over many slightly-mismatched renders
blurs). A small, well-chosen basis both bounds the cost and keeps the template
sharp.

The cap splits localization into two phases:

- **Phase A — congeal the basis.** Pick `K` views; run the existing congealing
  loop on them, unchanged (leave-one-out consensus, rounds, in-loop drop gates,
  convergence). `O(K²·n)` per round.
- **Phase B — register the tail.** Build one final all-basis consensus
  template (no holdout — tail views never contributed to it, so leave-one-out
  is unnecessary by construction) and run each remaining view's shift search
  against it **once** — no rounds, no Gram participation. `O((V−K)·n)` total,
  plus one small cache render per tail view.

Every observation is still localized and reported; only the *consensus
membership* shrinks. For `V ≤ K` (and `K = 0`) the path is bit-identical to the
uncapped implementation.

## Parameters

New fields on `KeypointLocalizeParams` (mirrored as PyO3 kwargs):

- `basis_max_views: u32` — consensus-basis cap `K`. `0` (default) disables the
  cap: all views congeal, exactly the current behavior. A non-zero `K` below `2`
  is raised to `2` — a leave-one-out consensus needs two members, and a
  one-member basis would leave the tail nothing to register against.
- `basis_force_track_views: bool` (default `true`) — reserve basis seats for
  the point's track views ahead of expansion candidates (they are the point's
  provenance and carry its detection keypoints). When the track alone exceeds
  `K`, the track views are themselves ranked by score and truncated at `K`.
- `basis_pick: BasisPick` — how the ranked candidate list fills the remaining
  seats: `TopScore` (default; the `K` best-scoring views) or `Strided` (every
  `ceil(m/K)`-th entry of the ranked list — trades per-view match quality for
  coverage of the ranked spectrum when the top scores cluster on
  near-duplicate frames).

## Basis ranking — score to the starting appearance

The basis wants the views that best match the point's **starting appearance
anchor**, ranked by windowed ZNCC:

1. **Caller-supplied scores** (preferred). The per-view scores are passed in
   parallel to the view sets (`view_scores`, same shape as `view_sets`; `NaN`
   = unscored). `sfm embed-patches` passes the `select_views` per-admitted-view
   `scores` straight through — each view's ZNCC against the point's track-view
   consensus reference, already computed during selection.
2. **Positional fallback.** With no caller scores, rank views by grazing angle
   (`|d̂·n̂|`, most frontal first) — the cosine the grazing pre-filter already
   computes per candidate, so the fallback is free. Deterministic; reached by
   callers that supply bare view lists, notably
   `sfm xform --localize-keypoints`.

Unscored (`NaN`) views rank below all scored views within their group. Track
membership is conveyed by the caller (`track_view_counts`, one integer per
point: the leading `t` entries of the point's view set are its track views —
matching the `select_views` output contract, whose `admitted` lists track views
first).

## Phase B mechanics

- **Final basis template.** After the basis loop exits, run one robust
  consensus build over the surviving basis members' final cores (the same IRLS
  as a congealing round, without a holdout) and materialize a single unit-norm
  template.
- **Tail cache.** Each tail view renders its context cache centered on its own
  seed offset (`render_context(au, av)` at the clamped seed), sized
  `R + 2·margin` — it searches one `±margin` window around the seed, so it
  needs no drift headroom (basis caches keep the `R + 4·margin` sizing).
- **Search + gates.** One shift search (same `search_strategy`) against the
  basis template. The existing per-view gates apply verbatim: drop when the
  refined keypoint moves `> max_shift_px` from the projection, or when the ZNCC
  falls below `min_relative_zncc ×` the **basis members'** median final ZNCC
  (the tail must meet the same relative bar the basis set). Kept tail views
  report their ZNCC in `loo_zncc` (for a tail view the reference is the basis
  template; the field keeps its name — it is still "this view against the
  consensus of the others").
- **Result contract.** `KeypointLocalization` keeps its shape: kept views (basis
  survivors + kept tail views) in the input view-set order, with keypoints,
  offsets, ZNCCs, and `rounds` (the basis round count). One field is added —
  `is_basis`, a per-kept-view flag (all `true` when the cap does not bite) —
  because the basis/tail split is otherwise unobservable, and the tail's ZNCC
  distribution is the quality signal the validation below reads.
- **No usable basis.** When the round loop collapses below two in-frame views
  there is no template to register against; the tail keeps its seed offsets with
  an unknown ZNCC, exactly what the loop already does for its own early exits.

## Plumbing

1. **`crates/sfmtool-core`** — `keypoint_localize/params.rs`: the three new
   fields + `BasisPick`. `keypoint_localize.rs`: basis pick before the render
   loop; phase-B registration after the round loop. Basis ranking helper with
   unit tests. Prof taps: a `basis_pick` phase, a `tail_register` phase, and
   `N_TAIL` / `N_BASIS` counters.
2. **`crates/sfmtool-py`** — `localize_keypoints`: kwargs `basis_max_views=0`,
   `basis_force_track_views=True`, `basis_pick="top_score"`, plus optional
   `view_scores` / `track_view_counts` inputs parallel to `view_sets`.
   `select_views` reports `track_view_count` per patch (how many leading
   `admitted` entries are track views), which is what feeds
   `track_view_counts`.
3. **Python pipeline** — `_embed_patches.py`: keep `selections[i]["scores"]`
   and the track-view counts (currently discarded) and pass both to
   `localize_keypoints`; `embed_patches(localize_basis_views=…)`.
4. **CLI** — `sfm embed-patches --localize-basis-views N` (default `0` =
   uncapped) and `sfm xform --localize-keypoints` `basis_max_views=N` option.
   Update `specs/cli/embed-patches-command.md` and the xform spec row.
5. **Cross-links** — `specs/core/patch-keypoint-localization.md` and
   `specs/core/keypoint-localization-search-cache.md` reference this file where
   they describe the consensus membership and cache sizing.

## Validation (A/B, required before changing the pipeline default)

Arms on a high-`V` reconstruction (expanded view sets reaching `V ≥ 100`) and a
moderate-`V` control (typical `V ≈ 15–40`):

| arm | question |
| --- | --- |
| `K=0` | baseline (bit-identity check for `V ≤ K` points included) |
| `K=12`, force-track, `TopScore` | proposed default |
| `K=12`, no force-track | do track views earn reserved seats? |
| `K=12`, `Strided` | does duplicate-frame clustering in the top scores hurt the tail fit? |
| `K=8`, `K=16` at the winner | sensitivity of the cap |

Metrics per arm, against `K=0`: localize wall time; per-observation keypoint Δ
(median / p99 source px); observation drop-set churn; tail-view ZNCC
distribution (a smeared or arc-biased basis template shows up as depressed tail
ZNCC on views far from the basis's viewpoints); and the downstream
embed→size-cull→BA+refine chain's reprojection / yield / rogue-normal metrics
for the candidate default.

### Measured (2026-07-26, DnDTabletop)

`sfmr/cleanup/gt-clean-01-ba.sfmr`, 337 images / 132,965 points, `--patch-size
5`, `resolution 24`, default `PlusDescent`. `select_views` produces a mean of
51.7 views/point, p99 236, max 323 — the high-`V` case the cap targets. The
moderate-`V` control is still outstanding; the `seoul_bull` fixture stands in
only as a `V ≤ K` no-op check in the test suite.

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

The pipeline default therefore stays `0`. The keypoint divergence is large
enough that adopting a non-zero default needs the downstream
embed→size-cull→BA+refine evidence this section calls for, which reprojection
error and yield can settle but these localizer-internal metrics cannot.

## Tests

- **Rust unit** (ranking helper): force-track reserves seats; oversized track
  ranked-and-truncated; `NaN` scores rank last; `Strided` picks the strided
  ranked entries; determinism.
- **Rust** (`keypoint_localize/tests.rs`): `basis_max_views=0` and `V ≤ K`
  bit-identical to the uncapped path; synthetic planted-offset scene — tail
  views recover their planted shifts against a `K`-basis template (accuracy
  bound shared with the existing congealing tests); tail gate drops a
  deliberately-mismatched tail view; result ordering/contract preserved.
- **Python** (`tests/`): kwargs accepted and threaded; embed pipeline run on
  the seoul_bull fixture with a small `K` succeeds, keeps observation parity
  with `K=0` on its (small-`V`) points, and `xform --localize-keypoints` with
  `basis_max_views` round-trips.

## Task-completion checks (from AGENTS.md)

- Rust: `pixi run cargo fmt && pixi run cargo clippy --workspace`.
- Bindings rebuilt for Python tests: `pixi run -e test maturin develop --release`.
- Python: `pixi run fmt && pixi run check`, then `pixi run test -- <modules>`.
- Rust tests: `pixi run cargo test --workspace`.
