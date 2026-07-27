# Keypoint localization — consensus-basis cap (basis congealing + tail registration)

> Status: **designed (2026-07-26), not yet implemented** — target:
> `crates/sfmtool-core/src/patch/keypoint_localize/` (params + orchestration),
> exposed as `PatchCloud.localize_keypoints(basis_max_views=…)` and threaded
> through `sfm embed-patches`. The Rust-layer default is `0` = off (bit-identical
> current behavior); the pipeline default is set by the A/B validation below.

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
  cap: all views congeal, exactly the current behavior.
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
2. **Stored-bitmap fallback.** When no scores are given and the cloud carries
   consensus `patch_bitmaps` (an already-embedded reconstruction), each view's
   score is one render + windowed ZNCC against the stored bitmap — `O(V·n)`,
   linear, computed inside the per-point localization.
3. **Positional fallback.** When neither exists, rank views by grazing angle
   (`|d̂·n̂|`, most frontal first). Deterministic and cheap; only reached by
   callers that supply bare view lists on un-embedded clouds.

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
- **Result contract.** `KeypointLocalization` is unchanged: kept views (basis
  survivors + kept tail views) in the input view-set order, with keypoints,
  offsets, ZNCCs, and `rounds` (the basis round count).

## Plumbing

1. **`crates/sfmtool-core`** — `keypoint_localize/params.rs`: the three new
   fields + `BasisPick`. `keypoint_localize.rs`: basis pick before the render
   loop; phase-B registration after the round loop. Basis ranking helper with
   unit tests. Prof taps: a `basis_pick` phase, a `tail_register` phase, and
   `N_TAIL` / `N_BASIS` counters.
2. **`crates/sfmtool-py`** — `localize_keypoints`: kwargs `basis_max_views=0`,
   `basis_force_track_views=True`, `basis_pick="top_score"`, plus optional
   `view_scores` / `track_view_counts` inputs parallel to `view_sets`.
3. **Python pipeline** — `_embed_patches.py`: keep `selections[i]["scores"]`
   and the track-view counts (currently discarded) and pass both to
   `localize_keypoints`; `embed_patches(localize_basis_views=…)`.
4. **CLI** — `sfm embed-patches --localize-basis-views N` (default from the A/B
   validation; `0` = uncapped) and `sfm xform --localize-keypoints`
   `basis_max_views=N` option. Update `specs/cli/embed-patches-command.md` and
   the xform spec row.
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
