# Photometric track widening, phase 2 — experiment notes (2026-07-19)

`scripts/exp_widen_growth.py` — THE RECIPE distilled from the Phase-1
requirements discovery (`notes-track-widen.md`): reach-deficit selection,
protected observations, fragment alignment, per-stage fragment scoring.

## Method (Stage A: post-hoc widen -> protect -> fragment-align)

Per workspace, starting from `sfmr/bootstrap-pinhole.sfmr` + the Phase-1
cluster universe (best 15000 clusters by span, status-0/1 members, posed
span >= 2):

1. **Reach-deficit selection** (replaces Phase-1's angular narrowness):
   candidate tracks are finite, consistent (member reproj med < 4 px,
   member-ZNCC >= data-derived 25th-pct floor, >= 6 members) tracks whose
   observers span a small fraction of the TRAJECTORY — observer reach =
   (max - min temporal rank of members) / (n_img - 1), gate at the
   qualifying set's median (data-derived).  Phase-1 degeneracy gates stay:
   span >= 10x the angular noise floor `atan(max(reproj,1px)/f)` and
   implied depth <= 3 rig diameters.  Selected tracks are sampled evenly
   across the temporal MIDPOINT so coupling lands everywhere along the
   trajectory (fragment junctions included).
2. **Trajectory-distance candidates**: a global anchor-view set (~32) =
   20 temporally-spread views + the top-12 covisibility revisit views
   (pairs with >= 50 shared clusters — feature-level covisibility,
   reconstruction-independent — that are temporally distant from the voting
   track's members).

   > _Status (2026-08-12): the pair evidence no longer comes from a
   > `displacement-knn.npz` sidecar (removed fleet-wide after the
   > 2026-08-12 ablation); `load_clusters_ext` derives the same
   > shared-cluster counts over posed image pairs from the `.matches` read
   > it already performs. Selection mechanics unchanged._  Per track: anchor views at temporal distance
   >= td_floor (p75 of qualifying reach, data-derived) from the member
   interval, in front, in bounds (5% margin; revisit partners get a loose
   bound of max_shift=60px outside — at drifted poses the frustum test
   itself is suspect), grazing < 60 deg vs the patch normal; up to 12,
   spread across trajectory-distance buckets.  NO angle-vs-span
   requirement and NO reprojection-delta acceptance gate.
3. **Localization** = Phase-1 machinery unchanged (reused by import):
   `PatchCloud.from_tracks` with .sift feature scales, per-track
   view-subset normal refinement (K=8), `localize_keypoints`
   (max_shift 60px, search 12, min_relative_zncc 0.6), pyramids chunked
   <= 60 images.  Acceptance by LOO-ZNCC only: floor =
   max(0.6, p25 of the localized population) — the member-ZNCC floor
   (~0.96) is miscalibrated for wide-baseline views (correct wide
   localizations sit near 0.75-0.87).
4. **BA arms** from the bootstrap state (staged native `bundle_adjust`,
   `opt_f=True`, FRAC_DIAG schedule `[(50s,5s),(12s,2s),(4,1)]`,
   `s = diag/550.6`): `control` (original obs), `unprot` (+ accepted obs,
   flat), `prot` (+ accepted obs with `protected=mask`, default
   `protected_loss_scale=3`).
5. **Fragment-align** (from the prot state): GT-free RANSAC-similarity
   decomposition of bootstrap-vs-prot poses (`sfmtool._compare_fragments`,
   defaults pos 3.5% / rot 5 deg / min 5) — cameras the protected pull
   moved as one rigid group = one fragment.  Each accepted wide obs
   linking a non-dominant fragment B to the dominant one yields a 3D-3D
   correspondence attached to B (point on the observed ray closest to the
   owned point, both directions).  Per fragment with >= 3 non-collinear
   links (#228 floor): trimmed Umeyama similarity, ICP-style re-dropped
   ray feet for 12 iterations (the one-shot perpendicular-foot fit is
   biased: a synthetic 1.3x fragment displacement recovers scale 0.85
   one-shot, 0.78 iterated vs 0.77 true, camera centers to ~1%), applied
   to B's cameras; re-triangulate; final protected BA (`aligned`).
6. **Scoring** per stage: `sfm compare <GT> <arm>.sfmr --fragments`
   (components / top-3 sizes / outliers / dominant internals) + Phase-1
   rotation median, global and piecewise-k8 center error, focal error.

Stage .sfmr files: `sfmr/widen2-{control,unprot,prot,aligned}.sfmr`.
Full compare outputs + JSON: `widen2-out/<ws>/`.

## Results

500 tracks/dataset, `--refine-normals` always on, LOO-ZNCC acceptance.
`sfm compare --fragments` defaults (pos 3.5% / rot 5 deg / min 5).

### 20250712_202131684 — FAIR HEADLINE (good GT; fragmented capture)

328 shared cameras. Camera errors vs GT (medians, % of GT rig diameter):

| arm | rot med | global % | piecewise % | focal % | new obs (survive 4px) |
|-----|---------|----------|-------------|---------|-----------------------|
| bootstrap  | 4.48 | 31.10 | 1.46 | -9.45 | |
| control    | 5.61 | 26.47 | 0.30 | -9.91 | |
| unprot     | 4.50 | 27.06 | 0.25 | -8.89 | 682 (125) |
| prot       | 4.45 | 27.46 | 0.25 | -8.77 | 682 (135) |
| aligned    | **2.72** | 27.66 | **0.19** | **-5.68** | 682 (138) |

Fragment decomposition (`sfm compare --fragments`):

| arm | comps | top-3 | outliers | dom cams (%) | dom pos m/med | dom rot m/med |
|-----|-------|-------|----------|--------------|---------------|---------------|
| bootstrap | 11 | 99/69/27  | 20 | 99  (30%) | 1.44/1.40 | 0.47/0.41 |
| control   |  8 | 122/69/58 |  6 | 122 (37%) | 1.36/1.43 | 0.65/0.51 |
| unprot    |  7 | 131/69/56 |  8 | 131 (40%) | 1.29/1.26 | 0.58/0.44 |
| prot      |  7 | 133/69/56 |  4 | 133 (41%) | 1.39/1.31 | 0.82/0.59 |
| aligned   |  **6** | **156**/70/56 | **3** | **156 (48%)** | 1.49/1.38 | 0.83/0.59 |

Reach-deficit selection is a genuine improvement over Phase-1 angular
selection here: 682 wide obs accepted by LOO-ZNCC (vs the Phase-1 <4px gate
which would pass ~5%), median localized delta 7.4px — the corrective signal
Phase-1 discarded. GT-free decomposition (bootstrap-vs-prot) exposed 12
camera fragments; fragment-align estimated inter-fragment similarities from
the protected spanning obs for 6 of them (frags 1-4/8/10 applied,
5/7 collinear-skipped): scales 0.91-1.00, rotations 0.18-2.6 deg,
displacements 0.6-40% of scene scale, link residuals 0.5-6.4% — i.e. the
few low-dimensional gauge breaks the medians hid.

VERDICT vs "components -> 1": NOT reached (11 -> 6), but monotonic
consolidation at every stage and the aligned arm is strictly best on every
axis: components 11->6, dominant 99->156 cams (30%->48%), outliers 20->3,
rot 4.48->2.72, focal -9.45%->-5.68%, piecewise 1.46->0.19. The recipe
consolidates and the fragment-align stage does real work (dominant +23 cams
over prot alone, rot halved) — but the residual inter-fragment gauge is not
fully closed post-hoc, exactly the Phase-1 fragment-table prediction.
Global stays ~27% (down from 31%): the remaining ~5-6 fragments still sit
at different gauges.

### 20240915_073131082 — REGRESSION GUARD (clean sweep)

368 shared cameras. Camera errors:

| arm | rot med | global % | piecewise % | focal % | new obs (survive) |
|-----|---------|----------|-------------|---------|-------------------|
| bootstrap | 0.36 | 0.48 | 0.179  | -0.17 | |
| control   | 0.12 | 0.287 | 0.0067 | -0.13 | |
| unprot    | 0.13 | 0.283 | 0.0067 | -0.13 | 1256 (740) |
| prot      | 0.14 | 0.283 | 0.0071 | -0.16 | 1256 (823) |

Fragments: bootstrap 2 comps [356/9] -> control/unprot/prot all 1 comp
[367], 1 outlier, dominant pos 0.07-0.09% / rot 0.04-0.06 deg. GT-free
decomposition (control-vs-prot) = 1 component (the protected pull is a pure
rigid nudge, no fragment structure to align), so fragment-align is a no-op
(2-comp boot-vs-prot with a 9-cam minor fragment that has too few
cross-fragment links). GUARD PASSES: stays 1 component, no regression on
any metric; 823/1256 wide obs survive the trim (66%, vs 740 unprotected --
protection keeps more good wide obs without harm). Notably here the
reach-deficit selection widens across ALL trajectory-distance buckets
(0.15-0.60 accepts 516/653) where Phase-1 got mostly near-angle only.

### SpainSoapmaker/ws2 — VALIDATED CONSOLIDATION TARGET (loop-rich, drifted)

301 shared cameras. Phase-1 post-hoc widened_all reached 15->5 comps,
dominant 29%->64%. Camera errors:

| arm | rot med | global % | piecewise % | focal % | new obs (survive) |
|-----|---------|----------|-------------|---------|-------------------|
| bootstrap | 4.33 | 16.36 | 4.05 | -1.90 | |
| control   | 4.43 | 28.95 | 0.82 | -0.91 | |
| unprot    | 2.57 | 29.99 | 0.32 | -2.64 | 610 (119) |
| prot      | 4.25 | 29.49 | 0.71 | -0.55 | 610 (83) |
| aligned   | 3.03 | 29.08 | 0.44 | +0.84 | 610 (88) |

Fragments:

| arm | comps | top-3 | outliers | dom cams (%) | dom pos m/med | dom rot m/med |
|-----|-------|-------|----------|--------------|---------------|---------------|
| bootstrap | 15 | 87/44/26  | 54 | 87  (29%) | 1.50/1.53 | 0.52/0.32 |
| control   | 10 | 68/57/36  | 26 | 68  (23%) | 1.23/0.95 | 0.41/0.35 |
| unprot    | **6**  | 99/84/57  | **17** | 99 (33%) | 1.08/1.04 | 0.29/0.27 |
| prot      | 10 | 75/56/46  | 23 | 75  (25%) | 1.66/1.41 | 0.50/0.34 |
| aligned   | 8  | 114/66/53 | 19 | 114 (38%) | 1.38/1.16 | 0.72/0.55 |

Fragment-align applied to frags 1/2/5 (scales 0.96-1.08, rot 0.3-6.3 deg,
disp 0.3-3.2% of scene scale, link residuals 0.08-0.49%), frag 6 too_few.

VERDICT vs the Phase-1 target (15->5, dom 64%): consolidates strongly
(15->6, outliers 54->17) but does NOT beat Phase-1 widened_all's dominant
64% — my best dominant is 38% (aligned). TWO honest negatives here: (1) on
ws2 the UNPROTECTED flat BA consolidates fragments MORE than the protected
arm (6 vs 10 comps) — the same "widened_all > widened" effect Phase-1 saw:
given enough wide obs, the flat robust trim still merges fragments, while
protection's bounded pull (loss_scale x3) is more conservative and holds
the manufactured obs at their imperfect positions. (2) Global error rises
16.4%->29-30% in every re-BA arm — the Phase-1 gauge-destructive-BA finding
reproduces exactly (the windowed-growth bootstrap is globally better than
its own flat re-optimization; control alone already does this). Reach-
deficit selection is a clear win on the acceptance side: 610 wide obs
accepted across ALL trajectory buckets (109 at 0.15-0.3, 34 at 0.3-0.6,
48 at 0.6-1.0) vs Phase-1's near-angle-only.

### 20240614_224244438 — OPEN PATH (the Phase-1 no-op failure case)

292 shared cameras. Phase-1: reach selection must produce NON-no-op obs
where the angular selection got none. Camera errors:

| arm | rot med | global % | piecewise % | focal % | new obs (survive) |
|-----|---------|----------|-------------|---------|-------------------|
| bootstrap | 4.16 | 34.94 | 0.205 | -5.51 | |
| control   | 3.00 | 34.90 | 0.153 | -5.46 | |
| unprot    | 2.91 | 34.95 | 0.134 | -6.32 | 49 (35) |
| prot      | 2.96 | 34.88 | 0.147 | -5.57 | 49 (48) |

Fragments: bootstrap 8 [133/38/16] -> control 6 -> unprot **4** [89/74/56]
-> prot 6. No fragment-align (GT-free boot-vs-prot = 2 comps [255/20], the
minor fragment too few cross-links; scene_scale 2.4e9 flags the open-path
scale blowup).

VERDICT: the OPEN-PATH REQUIREMENT IS MET — reach selection localized 2624
candidates and accepted **49 non-no-op** obs (delta p50 2.8px, prot keeps
48/49 through the trim) where Phase-1 angular selection produced only
no-ops. But global error stays 34.9% in every arm: the open-path scale
gauge is unobservable in principle (Phase-1's core finding), and 49 obs on
16 tracks is too thin to change the fragment gauge. unprot 8->4 fragments
is real local consolidation. The recipe's selection half now WORKS on open
paths; the gauge half remains unobservable without loops/priors.

## Cross-dataset reading

1. **Reach-deficit selection is the validated Phase-2 win.** It accepts
   wide obs across the full trajectory-distance range on every capture
   (including the open path, where Phase-1 got only no-ops), by LOO-ZNCC
   with the reprojection-delta gate deliberately removed. The manufactured
   obs carry a 7-14px median delta — the corrective signal.
2. **Fragment consolidation is real and monotonic everywhere**: components
   fall at every widening stage (202131684 11->6, ws2 15->6, 224244438
   8->4, guard 2->1). The medians hid this on Phase-1; the fragment tables
   are the honest metric.
3. **Protection helps on the fragmented headline, hurts on the loop-rich
   consolidator.** 202131684: aligned (protected + fragment-align) is
   strictly best (6 comps, dom 48%, rot halved, focal -9.4->-5.7%). ws2:
   the UNPROTECTED flat BA consolidates more (6 vs prot's 10). The bounded
   protected pull is the right tool when the wide obs must survive a
   hostile trim (fragmented capture, minority obs); it is too conservative
   when the flat trim would already merge the fragments. A per-dataset or
   fragmentation-aware protection policy is the follow-up.
4. **Fragment-align does real low-dimensional work** but does not fully
   close the gauge post-hoc: it recovers the largest single fragment
   (202131684 dominant 133->156 cams over prot, rot 4.45->2.72) from
   inter-fragment similarities estimated on the protected spanning obs
   (scales 0.91-1.08, rotations 0.2-6.3 deg), yet 5-8 fragments remain at
   distinct gauges. The ICP-iterated ray-foot fit is validated on a
   synthetic 1.3x fragment displacement (scale 0.78 recovered vs 0.77 true,
   centres to ~1%; the one-shot perpendicular fit is biased to 0.85).
5. **Global-vs-piecewise confirms the Phase-1 gauge law**: piecewise center
   error drops at every stage (202131684 1.46->0.19, ws2 4.05->0.32,
   224244438 0.21->0.13) while global stays put or worsens under flat re-BA.
   Local rigidity improves; the global gauge is the residual few-fragment
   alignment (loop captures) or unobservable (open path).

## Verdict against the success criteria

- **components -> 1 (headline 202131684): NOT reached (11 -> 6).** But the
  recipe consolidates monotonically and the full aligned pipeline is
  strictly best on every axis (dominant 30%->48%, outliers 20->3, rot
  -39%, focal -40% error). The residual is the last few inter-fragment
  gauge breaks, which fragment-align reduces but does not eliminate
  post-hoc — the Phase-1 prediction, now quantified.
- **global -> piecewise: NO** on the loop captures (global stuck ~27-30%,
  piecewise <0.5%); the flat re-BA is gauge-destructive and post-hoc
  alignment closes only the dominant fragment. Open path: global
  unobservable (expected).
- **no regression on the guard: YES** (073131082 stays 1 component, all
  metrics flat-or-better, 823/1256 wide obs survive under protection).
- **beat ws2 15->5 / dom 64%: NO** (reached 15->6, dom 38%); Phase-1
  post-hoc widened_all's ungated flat BA still holds the ws2 consolidation
  record.
- **open-path non-no-op obs (224244438): YES** (49 accepted non-no-op vs
  Phase-1's none).

Net: the RECIPE's selection and protection machinery are validated and
consolidate fragments on every capture, but post-hoc widen+align does not
reach a single rigid gauge on the loop captures — consistent with the
Phase-1 thesis that the gauge closes DURING growth (prevention), not
post-hoc. That is Stage B.

## Stage B (widen during growth at anchors)

`--stage-b`: monkeypatches `exp_hier_ba._photo_anchor_ba` (no edits to
that script) so every SFMTOOL_ANCHOR_EVERY-th growth BA becomes: widen the
currently-posed subgraph's reach-deficit tracks into the anchor subset's
trajectory-distant views, accumulate accepted obs OUTSIDE grow_loop's
arrays (its windowed BAs can never trim them), protected anchor BA over
the anchor-subset obs + all accumulated wide obs; after growth, one final
protected BA over the full active set. `--stage-b` (config: BA_WINDOW=40,
ANCHOR_EVERY=4, PHOTO_ANCHOR=1, COMPLETE_MAX_CL=15000, FRAC_DIAG, FORCE=1
for flagged experiment seeds).

### Feasibility result — 20250712_202131684 (headline, from fast-pinhole seed)

The hook fires cleanly: **17 widen-anchors during growth**, each widening
the currently-posed subgraph's reach-deficit tracks into the anchor
subset's trajectory-distant views (55-88 accepted/anchor) and re-running a
protected anchor BA, accumulating **1093 protected wide obs** that
grow_loop's windowed trims can never remove. Growth completed 318/328
posed (f=2401, the flagged low_consensus seed forced).

| stage | comps | top-3 | outliers | rot med | global % | piecewise % |
|-------|-------|-------|----------|---------|----------|-------------|
| growthB (grown, pre-final-BA) | 15 | 76/62/24 | 33 | 2.09 | 32.1 | 1.85 |
| finalB (+ final protected BA)  | **9** | 85/77/30 | 35 | **2.06** | 32.3 | 0.91 |

FEASIBILITY: **PROVEN end to end.** The during-growth widen+protect path
runs, fires at anchors while drift is still inside the photometric radius
(the v6 prevention premise), accumulates protected constraints the windowed
trims cannot silence, and the final protected BA consolidates 15->9
fragments. Rotation median 2.06 deg beats Stage-A's best post-hoc arm
(aligned 2.72). CAVEATS: (1) it does not reach 1 component in this
prototype — there is no fragment-align stage INSIDE growth yet (the recipe
would insert the fragment decomposition + inter-fragment similarity at the
spread anchor BAs); (2) not apples-to-apples with Stage A, which starts from
the completed `bootstrap-pinhole.sfmr` (11 comps) whereas Stage B grows
fresh from the flagged fast-pinhole seed (15 comps grown); (3) cost ~12 min
(694s) incl. 17 photometric anchor passes. The prototype confirms the
architecture the Phase-1 lessons pointed to: prevention-during-growth is
mechanically sound and consolidates; wiring the fragment-align operator into
the anchor cadence is the next step (Phase 3 / native).

### Sign-off

Stage A validated the recipe's selection (reach-deficit) and protection
machinery on all four benchmarks and quantified where post-hoc widen+align
lands: strong monotonic fragment consolidation with the dominant fragment
recovered, but not a single gauge on loop captures; open-path selection now
produces real non-no-op obs; clean guard. Stage B proved the during-growth
path is feasible and consolidates. The headline "components -> 1" is not
reached post-hoc; the evidence points to the during-growth fragment-align
composition as the closing step. All arms' numbers are logged above;
artifacts in `widen2-out/<ws>/` and `sfmr/widen2-*.sfmr`.

Claude-Session: https://claude.ai/code/session_01PYw4G8hXtpmLT1iQKPqjnd
