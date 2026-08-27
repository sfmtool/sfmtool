# Cluster Match Census

_Status: **implemented** — `sfmtool_core::analysis::cluster_census`, bound as
`sfmtool._sfmtool.analysis.cluster_census`. The score, the viewpoint groups,
the per-pair stats, and `sat_pct` are native and at parity with the Python
prototype (`scripts/seed_census.py`). The group-consistency companion
(§ [Group consistency](#companion-group-consistency)) is behind the opt-in
`compute_group_consistency`; unset, `CensusReport.group_consistency` is
`None`. The `census_echo` seed confidence flag is not wired up yet._

## Problem

A reconstruction can be internally consistent and wrong. Two failure shapes
share this property:

- **Misregistration (echo):** a viewpoint group — a covisibility-connected
  subset of the images — glued to the rest at a wrong relative pose: its own
  tracks reproject cleanly on both sides of the seam, and duplicate structure
  appears where the groups disagree.
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
  [cluster-warp-consistency.md](../patch/cluster-warp-consistency.md) — lower is
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
([cluster-covisibility.md](../features/cluster-covisibility.md)).

The solve's own track graph must not be used for grouping: tracks glued
across a bad seam span both sides by construction and inflate exactly the
cross-group counts the partition needs to be low. The raw covisibility graph
is independent of the solve.

Merge order is fixed by the gain `ΔQ = 2·(w_ab/2m − k_a·k_b/(2m)²)`; ties take
the **last** maximal pair over communities scanned in ascending `(a, b)` with
`a < b`, i.e. the merge maximizes `(ΔQ, a, b)` lexicographically. Group ids are
the positions of the communities in the live community list at the best-`Q`
partition (first partition to attain the maximum wins). Because the edge
weights are integer shared-cluster counts, the gains are exact and the whole
merge path is bit reproducible.

Fewer than two groups ⇒ the capture has no group structure to census; the
result is **unverifiable** (score 0 with `n_groups < 2`), which callers must
treat as "no evidence", not "clean".

### 2. Cluster placement at the candidate

Triangulate **every** raw cluster observed by ≥ 2 posed images at the
candidate poses (batch mid-point triangulation,
[batch-triangulation-api.md](../reconstruction/batch-triangulation-api.md)), then compute:

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
([cluster-warp-consistency.md](../patch/cluster-warp-consistency.md)) — how well its
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

### <a name="companion-group-consistency"></a>6. Companion: group consistency

The census score reports **how much** cross-group evidence the candidate
leaves unsatisfied; the group-consistency companion asks whether that
disagreement is **coherent** — explainable by group-level pose error — and
estimates it. Jointly over all groups (the largest group by posed-image
count fixes the gauge, ties to the lowest group id), estimate the
per-group 7-dof similarities
(rotation, translation, log scale) that minimize a robust cost over the
eligible bridges: a global pose-consistency solve over the viewpoint-group
graph, the group-level analogue of the per-cluster census.

A correction `(Q, t, s)` acts on its group's content as the world
similarity `W(x) = s·Q·x + t`, equivalently on its cameras as

```
R' = R·Qᵀ        C' = s·Q·C + t
```

which leaves that group's own projections untouched — its structure moves
with it — and changes only where its rays meet the other groups'. Bridges
are therefore **re-triangulated** at the corrected poses and re-scored;
there is no fixed structure to hold on to. The cost is the soft-L1 sum over
the bridges' per-observation pixel residuals, minimized by
Levenberg–Marquardt from the identity correction over 7 × (n_groups − 1)
parameters — small enough for dense normal equations and a
central-difference Jacobian of the triangulate-and-project chain. A
bridge is re-triangulated from its own observations alone, so perturbing
one group's parameter block changes nothing for a bridge with no
observation on that group's images: each finite difference re-evaluates
only the bridges its block moves. The
robust loss is what lets the fit run on the eligible bridges directly,
false matches that survived the screen included; an observation the
corrected placement pushes outside the camera model's domain is charged a
large bounded residual — never the cheap way out, but one crossing cannot
outweigh the population. Translation parameters are scaled by the scene
radius (the median camera distance from the component-wise median of the
camera centers) so all seven parameters of a block are comparable. Report:

- the per-group **corrections** — identity corrections mean the candidate
  is already group-consistent, for every group with eligible bridge
  evidence (a group with none is unconstrained and carries the identity by
  construction; its absence from the pair list identifies it);
- **explained** (`explained_pct`, with its numerator `n_explained` and
  denominator `n_unsatisfied_before`) — the percent of
  previously-unsatisfied high-parallax bridges satisfied at the corrected
  placements. The ratio is unshrunk, so consumers should read the
  denominator with it: 2 of 2 explained is not 300 of 300;
- the **net** change in the total number of satisfied bridges — the joint
  solve is scored on **all** bridges, so a correction that fixes one seam
  by breaking another nets ≈ 0 and does not count as an explanation.

The decomposition discriminates a flag's cause: a genuine misregistration
is coherent (a non-trivial correction explains its seam's unsatisfied
bridges); a false-match population that survived the eligibility screen is
incoherent (jointly unsatisfiable by any rigid correction — explained
fraction ≈ 0). The corrections are the natural initialization for callers
that re-glue a flagged group — after checking the net, which can come out
*negative* when the group split does not align with the actual
misplacement; the operation itself is analysis only and
never modifies the candidate, and it costs a solve, so it is opt-in
(`compute_group_consistency`) and leaves every other field of the report
untouched.

It reports **nothing** — not a vacuous identity — where it has nothing to
say: fewer than two groups (no group structure), or no eligible measurable
bridge (no cross-group evidence). A solve that cannot descend (singular
normal equations, no admissible step) reports the identity corrections it
started from, which score explained fraction 0 with the net unchanged.

## Outputs

```
CensusReport {
    score:      f64,              // max per-pair Wilson lower bound
    n_groups:   usize,            // < 2 ⇒ unverifiable, score is vacuous
    group_of:   Vec<i32>,         // per input image, -1 = unposed
    pairs:      Vec<PairStats>,   // (ga, gb, n_eligible_hi, n_unsatisfied_hi, wilson_lb)
    sat_pct:    f64,
    group_consistency: Option<GroupConsistency>,
        // per-group corrections; explained_pct with n_explained /
        // n_unsatisfied_before; net_before / net_after;
        // None unless compute_group_consistency, and where § 6 declines
}
```

`pairs` carries every group pair joined by at least one bridge cluster,
including pairs whose bridges are all ineligible or low-parallax
(`n_eligible_hi == 0`, `wilson_lb == 0`) — the absence of evidence for a seam is
itself reportable. `sat_pct` is computed whenever there are eligible, measurable
clusters, including the `n_groups < 2` case where the score is vacuous: it
needs no grouping.

The candidate's intrinsics enter as a full camera model, not a bare focal, so
the operation applies to any model the projection supports; the arbitration
callers pass a shared pinhole.

## Callers

- **Finalization focal arbitration** (`_finalize_seed`): score each candidate
  BA result; keep the lower-scoring candidate, ties to the vote.
- **`census_echo` seed confidence flag**: after finalization, flag the seed
  when `score ≥ flag_threshold` (and, with § 6 enabled, explained fraction ≥
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
| `compute_group_consistency` | false | opt-in for § 6; it costs a solve and answers a different question from the score |
| § 6 robust scale | 3.0 px | soft-L1 transition of the group-consistency cost; above `sat_px`, so a satisfied bridge sits in the quadratic regime and a false match cannot drag the solve |
| § 6 fit-set cap | 1200 bridges | 7 dof per group are over-determined by a few hundred bridges, so beyond the cap the fit strides down to it; the fit set bounds the whole descent — Jacobian, normal equations, trial steps — and a finite difference within it touches only the fit bridges its parameter block moves, while the corrections are still *scored* on the complete bridge population |

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
not), segmented per-cluster medians, and the Wilson bound. The group
consistency of § 6 adds a small robust LM solve (7 × (n_groups − 1)
parameters) over re-triangulated bridge residuals. Home: `sfmtool-core`
alongside the covisibility and triangulation kernels, exposed through
`sfmtool-py` as `analysis.cluster_census(...)`.

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
