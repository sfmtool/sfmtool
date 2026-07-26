# Cluster Match Census

_Status: **prototyped** (Python: `scripts/seed_census.py`, in-process scorer used
by the seed finalization's focal arbitration). This spec is the design for the
`sfmtool-core` operation, its Python binding, and the `census_echo` seed
confidence flag. The group-consistency companion
(§ [Group consistency](#companion-group-consistency)) is prototyped in a
worst-pair variant alongside but specified here as a second phase._

## Problem

A reconstruction can be internally consistent and wrong. Two failure shapes
share this property:

- **Misregistration (echo):** a viewpoint group — a covisibility-connected
  subset of the images — glued to the rest at a wrong relative pose — its own tracks reproject cleanly on both sides of the
  seam, and duplicate structure appears where the groups disagree.
- **Focal compensation (bas-relief):** poses and structure bend smoothly to
  absorb a wrong focal; every retained track fits.

Reprojection error over the solve's **own tracks cannot detect either**: the
solve chose those tracks, and both failure modes are (locally) optimal for
them. The evidence that does discriminate is the raw cluster set the solve did
**not** consume: high-parallax clusters whose members span distinct
viewpoint groups constrain the groups' relative placement, and a solve that
placed a group wrongly — or bent under a wrong focal — necessarily leaves a
fraction of them unsatisfiable. No individual cluster can be trusted — any
one may be a false match — which is why the operation is a **census**: the
population is screened for eligibility (§ 3) and the score is the
unsatisfied *fraction* under a confidence bound (§ 4), never a verdict on a
single correspondence.

Because the score is computed from evidence **outside** the solve's objective,
it can arbitrate between candidate solves of the same capture (two focal
candidates, a placement and its correction) where their internal residuals
cannot — both candidates fit their own tracks; only one fits the withheld
cross-group evidence.

## Inputs

- The raw patch clusters of the workspace (`.matches` clusters backbone):
  flat observation arrays `(cluster, image, uv)`, plus a per-cluster
  **warp-consistency residual** (the worst finite member residual of the
  matching-time consistency fit,
  [cluster-warp-consistency.md](cluster-warp-consistency.md) — lower is
  better).
- A candidate solve of a subset of those images: world→camera rotations,
  camera centers, and a shared pinhole focal (canonical frame, −Z forward).

The candidate's own tracks, points, and residuals are **not** inputs.

The evidence unit is the multi-view **cluster match**:

- a cluster carries **one warp-consistency residual for the whole
  correspondence**, which is what the eligibility screen
  thresholds on; pairwise matches have no per-correspondence multi-view
  consistency signal;
- a cluster's **≥ 2-image span** lets a single correspondence bridge a group
  pair with measurable triangulation parallax; a pairwise match constrains
  only its own pair;
- a cluster is **one physical point**, so the census triangulates and judges
  each piece of evidence once, instead of re-deriving agreement from a web of
  pairwise links.

## Algorithm

### 1. Viewpoint groups

Partition the posed images into groups by **greedy-modularity communities**
(Clauset–Newman–Moore, CNM; deterministic, best-Q partition over the merge
path) of the raw cluster-covisibility graph — shared-cluster counts
restricted to the posed set, one count per (cluster, pair), via
`ClusterCovisibility`
([cluster-covisibility.md](cluster-covisibility.md)).

The solve's own track graph must not be used for grouping: tracks glued
across a bad seam span both sides by construction and inflate exactly the
cross-group counts the partition needs to be low. The raw covisibility graph
is independent of the solve.

Fewer than two groups ⇒ the capture has no group structure to census; the
result is **unverifiable** (score 0 with `n_groups < 2`), which callers must
treat as "no evidence", not "clean".

### 2. Cluster placement at the candidate

Triangulate **every** raw cluster observed by ≥ 2 posed images at the
candidate poses (batch mid-point triangulation,
[batch-triangulation-api.md](batch-triangulation-api.md)), then compute:

- `med(c)` — the per-cluster **median** reprojection residual of its
  observations at the candidate (median, not mean: a cluster is "satisfied"
  when the candidate can explain most of its members; single outlier members
  are matching noise).
- `para(c)` — triangulation parallax: the maximum angle between observation
  rays to the triangulated point.

A cluster is **measurable** at the candidate when at least two of its
observations fall on posed images and both the triangulated point and
`med(c)` are finite — the candidate can place and score it. Only
measurable clusters are considered further; a measurable cluster is
**satisfied** when `med(c) < sat_px`.

### 3. Evidence eligibility (data-derived)

False matches (repeated texture gluing different physical points into one
cluster) must not dominate the census population. Each cluster carries its
matching-time **warp-consistency residual**
([cluster-warp-consistency.md](cluster-warp-consistency.md)) — how well its
members mutually agree under the fitted patch warps; a genuine physical
correspondence has a low residual, a phantom one usually does not. The
eligibility threshold on that residual is derived from the candidate
itself: clusters the candidate **satisfies** are overwhelmingly genuine (a
false match has no reason to reproject consistently at any solve), so their
warp-consistency distribution calibrates what genuine correspondences look
like on this capture:

```
q_eligible    =  P95( warp_consistency(c) | c satisfied )
eligible(c)  ⇔  warp_consistency(c) finite  ∧  warp_consistency(c) ≤ q_eligible
```

This adapts per capture — a clean capture gets a tight threshold, a noisy
one a loose one — with no global constant.

Eligibility is a population screen, not a per-cluster certification: an
eligible cluster may still be false, and a genuine cluster can fall outside
the P95 tail. The census is built to absorb that — the score is a fraction
over the eligible population under a Wilson bound, so residual
contamination shifts it marginally rather than flipping it.

### 4. Per-pair census

A **bridge** is a measurable cluster whose observations span ≥ 2 groups. For each
group pair, over the bridges of that pair that are `eligible` and
high-parallax (`para ≥ hi_para`):

```
frac(pair)   =  #unsatisfied / #eligible-high-parallax
census(pair) =  WilsonLB( #unsatisfied, #eligible-high-parallax, z )
```

The Wilson lower bound shrinks small denominators toward zero: three
unsatisfied bridges out of four is suspicion, not certainty. Scoring **per
pair** (not pooled) prevents a fine partition from diluting one bad seam with
the satisfied bridges of good seams.

```
census_score  =  max over group pairs of census(pair)
```

High-parallax gating is what makes the score respond to the failure modes:
low-parallax bridges tolerate large relative-placement and focal error (their
rays barely converge), so they carry count but no constraint. The residual of
a high-parallax bridge grows with the seam's gauge disagreement and with
focal error, which is why the score decreases monotonically as a candidate
approaches the true placement and focal — the property the arbitration
callers rely on.

### 5. Companion: global satisfaction

```
sat_pct  =  % of all eligible, measurable clusters satisfied
```

A globally-deformed solve (every seam slightly wrong, no single worst pair)
degrades `sat_pct` without producing a large pairwise census. Callers that
gate should test both.

### <a name="companion-group-consistency"></a>6. Companion: group consistency (phase 2)

The census score reports **how much** cross-group evidence the candidate
leaves unsatisfied; the group-consistency companion asks whether that
disagreement is **coherent** — explainable by group-level pose error — and
estimates it. Jointly over all groups (the largest group fixes the gauge),
estimate the per-group 7-dof similarities (rotation, translation, log
scale) that minimize a robust cost over the eligible bridges: a global
pose-consistency solve over the viewpoint-group graph, the group-level
analogue of the per-cluster census. Report:

- the per-group **corrections** — identity corrections mean the candidate
  is already group-consistent;
- the **explained fraction** — previously-unsatisfied high-parallax
  bridges satisfied at the corrected placements;
- the **net** change in the total number of satisfied bridges — the joint
  solve is scored on **all** bridges, so a correction that fixes one seam
  by breaking another nets ≈ 0 and does not count as an explanation.

The decomposition discriminates a flag's cause: a genuine misregistration
is coherent (a non-trivial correction explains its seam's unsatisfied
bridges); a false-match population that survived the eligibility screen is
incoherent (jointly unsatisfiable by any rigid correction — explained
fraction ≈ 0). The corrections are the natural initialization for callers
that re-glue a flagged group; the operation itself is analysis only and
never modifies the candidate.

## Outputs

```
CensusReport {
    score:      f64,              // max per-pair Wilson lower bound
    n_groups:   usize,            // < 2 ⇒ unverifiable, score is vacuous
    group_of:   Vec<i32>,         // per input image, -1 = unposed
    pairs:      Vec<PairStats>,   // (ga, gb, n_eligible_hi, n_unsat_hi, wilson_lb)
    sat_pct:    f64,
    // phase 2:
    group_consistency: Option<GroupConsistency>,
        // per-group corrections, explained fraction, net
}
```

## Callers

- **Finalization focal arbitration** (`_finalize_seed`): score each candidate
  BA result; keep the lower-scoring candidate, ties to the vote.
- **`census_echo` seed confidence flag**: after finalization, flag the seed
  when `score ≥ flag_threshold` (and, with phase 2, explained fraction ≥
  a coherence threshold to suppress junk-evidence flags). The flag reports the
  failure axis the focal flags cannot see: correct focal, wrong placement.
- **Fleet / analysis tooling**: per-solve echo screening over a workspace.

## Parameters

| name | default | nature |
|---|---|---|
| `sat_px` | 2.0 px | the pipeline's shared inlier threshold (same constant as the BA inlier accounting); candidates for a resolution-relative form should change it everywhere together |
| `hi_para` | 5° | well under the parallax of genuine cross-group bridges (tens of degrees) and above the regime where residuals stop responding to gauge error |
| warp-consistency percentile | P95 | tail width of the eligibility threshold; the threshold itself is data-derived, the tail is fixed |
| Wilson `z` | 1.96 | standard 95 % bound |
| `flag_threshold` | 0.25 | calibration constant for the flag caller; **not yet data-derived** — revisit with a per-capture null (e.g. the census of a within-group split, which should be ≈ 0) |

## Blind spots (by design; callers must know)

1. **Phantom-built seams are invisible.** A seam glued by predominantly false
   matches has no eligible bridges to census — the eligibility screen
   removes the only evidence. Detecting those requires the inverse
   lens (many high-count, poor-warp-consistency bridges gluing groups), a
   separate operation.
2. **Sub-community seams evade the pair census.** A misregistered pair of
   frames absorbed inside one community contributes no bridge. Hierarchical
   grouping (census within each group recursively) is the extension.
3. **No high-parallax bridges ⇒ unverifiable.** Report it as such; a guard
   that maps this to "pass" silently certifies exactly the captures it cannot
   see.

## Core promotion notes

The operation composes existing native kernels — `ClusterCovisibility`
counts, batch triangulation, per-image reprojection residuals — plus three
small new pieces: CNM modularity grouping (n ≤ a few hundred posed images;
O(n³) naive is acceptable, the Python prototype's per-merge recompute is
not), segmented per-cluster medians, and the Wilson bound. Phase 2's group
consistency needs a small robust LM solve (7 × (n_groups − 1) parameters)
over re-triangulated bridge residuals. Natural home: `sfmtool-core` alongside the covisibility and
triangulation kernels, exposed through `sfmtool-py` as
`analysis.cluster_census(...)`.

## Evidence

Fleet prototype (37-dataset seed campaign, workspace-prep `census_one.py` /
`echo-census-findings.md`): flags all five substantial human-confirmed echo
seeds (one detectable **only** by the census — 92 % of its matches globally
satisfied); passes 6 of 7 human-trusted good solves; exposed five stale
"clean" labels later confirmed as hidden seed defects. Score tracked focal
error monotonically on the approved DnDTabletop GT (0.301 at +36 % focal,
0.254 at +10 %, 0.216 at +0.4 %), and as a dual-candidate arbiter in the seed
finalization chose correctly on 10 of 12 flagged fleet datasets (the two
misses: one junk-dominated capture inside the phantom blind spot, one
0.03-margin tie). Pose-error validation (`seedval-report.tsv`): census ≥ 0.7
on every "ok"-flagged seed with median center error > 9 % of scene radius,
except the two phantom-seam captures (blind spot 1). The prototype's
worst-pair variant of the group-consistency solve separated junk flags
from genuine misregistrations across the fleet (explained fraction
0–5 % vs 49–97 %).
