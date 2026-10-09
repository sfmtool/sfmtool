# Keypoint localization — consensus-basis cap (basis congealing + tail registration)

## Motivation

Keypoint localization refines one 3D point's image position in every view that
sees it, aligning those views against a shared consensus of what the point looks
like. The consensus-basis cap bounds how many views take part in *building* that
consensus — a small, well-chosen basis — and registers every remaining view once
against the finished template. It exists because the cost of the consensus grows
with the square of the view count while its quality stops improving after a
modest number of well-matched views.

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
membership* shrinks. With `K = 0` — and whenever the point's **candidate count**
does not exceed `K` — the path is bit-identical to the uncapped implementation.
The candidate count is what survives the grazing and projection pre-filters and
the view-set dedup, not the raw `view_set` length: the pick runs on that list,
so a point with 20 raw views of which 10 survive the filters is uncapped at
`K = 12`.

## Parameters

The cap is orchestrated in
[keypoint_localize.rs](../../../crates/sfmtool-core/src/patch/keypoint_localize.rs)
with the ranking pick in
[basis.rs](../../../crates/sfmtool-core/src/patch/keypoint_localize/basis.rs),
exposed as `PatchCloud.localize_keypoints(basis_max_views=…)`,
`sfm embed-patches --localize-basis-views` and
`sfm xform --localize-keypoints basis_max_views=…`. The default is `8` at every
layer; `0` restores the uncapped path, which has the cleanest error and is the
one to reach for when producing ground truth.

New fields on `KeypointLocalizeParams` (mirrored as PyO3 kwargs):

- `basis_max_views: u32` — consensus-basis cap `K`, default `8`. `0` disables the
  cap: all views congeal, exactly the current behavior. A non-zero `K` below `2`
  is raised to `2` — a leave-one-out consensus needs two members, and a
  one-member basis would leave the tail nothing to register against.
- `basis_force_track_views: bool` (default `true`) — reserve basis seats for
  the point's track views ahead of expansion candidates (they are the point's
  provenance and carry its detection keypoints). When the track alone exceeds
  `K`, the track views are themselves ranked by score and truncated at `K`.
- `basis_pick: BasisPick` — how the ranked candidate list fills the remaining
  seats: `TopScore` (default; the best-scoring entries) or `Strided` (every
  `ceil(m/s)`-th entry — trades per-view match quality for coverage of the
  ranked spectrum when the top scores cluster on near-duplicate frames). `m`
  and `s` are the list and seat count the pick is actually filling: with
  `basis_force_track_views` the track views take their seats first by score and
  the stride then runs over the **non-track remainder** with the seats that are
  left, so `m` is the non-track candidate count and `s = K − (seats the track
  took)`. Without the reservation `m` is the whole candidate list and `s = K`.
  If the stride runs off the end before the seats are full it tops up in rank
  order.

Only `basis_max_views` is reachable from the CLI (`sfm embed-patches
--localize-basis-views`, `sfm xform --localize-keypoints basis_max_views=`);
`basis_force_track_views` and `basis_pick` are binding-level knobs on
`PatchCloud.localize_keypoints`, since neither moved a measurable metric (see
[Why the default is eight views](#why-the-default-is-eight-views)).

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
  basis template — preceded, as in the basis path, by the member self-similarity
  gate on the tail view's own tile at its seed, so a view that pins no 2D
  position is never even searched. The remaining per-view gates apply verbatim:
  drop when the refined keypoint moves `> max_shift_px` from the projection, when
  the ZNCC is finite and below `min_absolute_zncc`, or when it falls below
  `min_relative_zncc ×` the **basis members'** median final ZNCC. There is no
  two-view floor here: the basis already carries the point, so a failing tail
  view is simply not registered.
  That is the same *threshold rule* the round loop applies, but not the same
  measurement: a basis member's ZNCC is against a leave-one-out consensus of the
  other members, a tail view's against the no-holdout template of all of them,
  and a sharper reference scores lower for the same quality of fit. Kept tail
  views report their ZNCC in `loo_zncc` (the field keeps its name — it is still
  "this view against the consensus of the others").
- **Mixed channel counts.** The template lives in the channel space common to
  the *basis* caches. A tail view can be narrower (a grayscale frame among
  colour ones); it is then scored over the channels it has, which is the round
  loop's own rule ("score in the space common to the participating views")
  applied pairwise. Only trailing channels drop out, so the template needs no
  rebuild. A tail view left with no scored channel is unscorable and is dropped
  by the gates, like one whose window is out of frame.
- **Result contract.** `KeypointLocalization` keeps its shape: kept views (basis
  survivors + kept tail views) in the input view-set order, with keypoints,
  offsets, ZNCCs, and `rounds` (the basis round count). One field is added —
  `is_basis`, a per-kept-view flag (all `true` when the cap does not bite) —
  because the basis/tail split is otherwise unobservable, and the tail's ZNCC
  distribution is a quality signal the
  [measurements](keypoint-localization-consensus-basis-measurements.md) read.
- **No usable basis.** When the round loop collapses below two in-frame views,
  or leaves no textured channel, there is no template to register against and
  the tail keeps its seed offsets with an unknown ZNCC. The agreement gates
  cannot be evaluated without a template (and an unknown ZNCC is not "finite and
  below" the absolute floor either), but the positional one still is: a
  seed can already sit further than `max_shift_px` from the projection and
  nothing downstream would catch it. The member self-similarity gate needs a
  rendered tile and this path renders none — that render is the cost it exists
  to avoid — so it does not apply here. This is *not* the same as the round loop's
  own early exits, whose survivors have at least been read in frame and, past
  round 1, already faced both gates — so `N_TAIL_NO_BASIS` reports how many tail
  views took it (244 of 475,645 on the measured capture).

## How the ranking inputs reach the kernel

The pick needs two things the kernel cannot derive for itself: a per-view score
to rank by, and how many of a point's views are its own track views (so those can
be reserved). Both are optional inputs carried down from view selection.

`PatchCloud.localize_keypoints`
([localize_keypoints.rs](../../../crates/sfmtool-py/src/patches/localize_keypoints.rs))
takes `basis_max_views`, `basis_force_track_views` and `basis_pick` alongside two
optional maps parallel to `view_sets`: `view_scores`, a `point_index -> [score,
…]` mapping, and `track_view_counts`, a `point_index -> t` mapping. `select_views`
supplies both — it reports a `track_view_count` per patch, the number of leading
`admitted` entries that are track views — and
[`_embed_patches.py`](../../../src/sfmtool/_embed_patches.py) threads them
through from the selection it already holds. With no scores the ranking falls
back to grazing angle, which is why an empty score slice and no scores at all
must rank identically.

`SFMTOOL_PROFILE` reports the cap's own cost and reach: `basis_pick` and
`tail_register` are separate profiling phases, and `N_BASIS` / `N_TAIL` /
`N_TAIL_NO_BASIS` count the views that took each path
([prof.rs](../../../crates/sfmtool-core/src/patch/keypoint_localize/prof.rs)).

## Why the default is eight views

The cap was measured on a high-`V` capture (DnDTabletop: 337 images, 132,965
points, a mean of 51.7 views per point after `select_views`, p99 236). The
arms, tables and readings are in
[keypoint-localization-consensus-basis-measurements.md](keypoint-localization-consensus-basis-measurements.md).
What they show:

- The cap removes the quadratic consensus terms. Uncapped they take 43 % of the
  localization pass; at `K = 8…16` they take 2–5 %. The pass runs about 2.6×
  faster, and `sfm embed-patches` about 2× (21 minutes uncapped, 10–11
  minutes capped). The speedup is about the same across
  `K = 8…16`, because what remains is per-view work: every view still renders a
  cache and runs at least one search.
- Capped runs keep about 1.7 % more points and 4.9 % more observations, because
  a tail view faces the relative-ZNCC gate once against a finished template
  instead of across rounds whose consensus, and so whose bar, moves.
- The cost is a small rise in error after `--filter-by-patch-size 3.0` and
  `--bundle-adjust --refine-normals --refine-keypoints`. On the points common
  to all arms, the median reprojection error is 1.365 px at `K = 8` and 1.351 px
  at `K = 16` against 1.307 px uncapped, and the share of normals more than 70°
  off their 8-NN consensus is 4.25 % and 4.08 % against 3.96 %.
- `K = 16` is slightly better than `K = 8` on every downstream metric, by small
  margins. `K = 8` does the least work and builds the sharpest basis template
  (median basis ZNCC 0.972 against 0.968 at `K = 16`), so it is the default.
- `basis_force_track_views` changes no metric by more than 0.5 %, and
  `Strided` improves on `TopScore` by less than raising `K` from 12 to 16 does.
  Both stay binding-level knobs; the track reservation is on for provenance.

`K = 0` is the choice where error metrics are the product, such as ground-truth
cleanup. The cap has not been measured on a moderate-`V` capture
(`V ≈ 15–40`), and no run has separated keypoint quality from the extra
observations the single tail gate keeps.

## Tests

- **Rust unit** (ranking helper): force-track reserves seats; oversized track
  ranked-and-truncated; `NaN` scores rank last; `Strided` picks the strided
  ranked entries; determinism.
- **Rust** (`keypoint_localize/tests.rs`): `basis_max_views=0` and `V ≤ K`
  bit-identical to the uncapped path; synthetic planted-offset scene — tail
  views recover their planted shifts against a `K`-basis template (accuracy
  bound shared with the existing congealing tests); tail gate drops a
  deliberately-mismatched tail view; result ordering/contract preserved.
- **Rust** (mixed channels): a tail view narrower than the basis template's
  channel space is scored over its own channels, not indexed out of bounds; the
  same scene through the uncapped path is unchanged.
- **Rust** (no-basis path): with no template built, an un-registered tail view
  survives a loose `max_shift_px` and is dropped by a tight one.
- **Rust** (empty scores): an empty per-point score slice ranks by grazing
  angle, identically to supplying no scores at all.
- **Python** (`tests/`): kwargs accepted and threaded; the embed pipeline run on
  the seoul_bull fixture with a small `K` lands within 5 % of the `K=0` run's
  point and observation counts (its view sets are mostly at or under the cap, so
  most points take the uncapped path outright); `xform --localize-keypoints`
  with `basis_max_views` round-trips; and a whole-cloud `view_scores` map drives
  chunked `point_indexes` calls to the same result as one shot.
